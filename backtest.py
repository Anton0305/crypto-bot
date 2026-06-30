"""
Backtest для Crypto Signals Bot v4.0
Проверяет как сигналы работали бы за последний месяц.

Логика:
1. Скачиваем 60 дней часовых свечей по каждой паре
2. "Прокручиваем" время вперёд час за часом
3. На каждом часе считаем сигнал (как в реальном боте)
4. Если сигнал BUY/SELL — смотрим что случилось дальше:
   достигнут TP1? TP2? или сработал SL?
5. Считаем статистику: винрейт, средний R/R, какие индикаторы
   чаще встречаются в выигрышных сигналах
"""

import asyncio
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import aiohttp

BINANCE_BASE = "https://api.binance.com/api/v3"

WATCHLIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","ENAUSDT",
    "AVAXUSDT","DOTUSDT","LINKUSDT","ARBUSDT","OPUSDT",
    "INJUSDT","SUIUSDT","TIAUSDT","WIFUSDT","PENDLEUSDT",
]

# ── DATA FETCH ────────────────────────────────────────────────────────────────

async def fetch_klines_full(session, symbol, interval="1h", days=60):
    """Качаем длинную историю с пагинацией (Binance отдаёт максимум 1000 за раз)"""
    limit = 1000
    all_data = []
    end_time = None

    needed = days * 24 + 200  # с запасом для индикаторов
    while len(all_data) < needed:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time
        async with session.get(f"{BINANCE_BASE}/klines", params=params,
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.json()
        if not isinstance(data, list) or len(data) == 0:
            break
        all_data = data + all_data
        end_time = data[0][0] - 1
        if len(data) < limit:
            break
        await asyncio.sleep(0.2)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df.set_index("time", inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

# ── INDICATORS (та же логика что в bot_v4.py) ─────────────────────────────────

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
    rsi = calc_rsi(series, period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def calc_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def calc_bollinger(series, period=20):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + 2*std, mid, mid - 2*std

def calc_williams_r(df, period=14):
    high_max = df["high"].rolling(period).max()
    low_min  = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)

def calc_cci(df, period=20):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - sma) / (0.015 * mad + 1e-10)

def calc_obv(df):
    direction = np.sign(df["close"].diff().fillna(0))
    return (df["volume"] * direction).cumsum()

def calc_vwap(df):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()

def calc_ichimoku(df):
    h9  = df["high"].rolling(9).max()
    l9  = df["low"].rolling(9).min()
    tenkan = (h9 + l9) / 2
    h26 = df["high"].rolling(26).max()
    l26 = df["low"].rolling(26).min()
    kijun = (h26 + l26) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    h52 = df["high"].rolling(52).max()
    l52 = df["low"].rolling(52).min()
    senkou_b = ((h52 + l52) / 2).shift(26)
    return tenkan, kijun, senkou_a, senkou_b

# ── PRECOMPUTE ALL INDICATORS ──────────────────────────────────────────────────

def precompute(df):
    """Считаем все индикаторы один раз на всём датасете (быстрее чем по кускам)"""
    close = df["close"]
    ind = {}
    ind["rsi"] = calc_rsi(close)
    ind["macd"], ind["macd_sig"], ind["macd_hist"] = calc_macd(close)
    ind["ema9"]  = close.ewm(span=9,  adjust=False).mean()
    ind["ema21"] = close.ewm(span=21, adjust=False).mean()
    ind["ema50"] = close.ewm(span=50, adjust=False).mean()
    ind["bb_up"], ind["bb_mid"], ind["bb_low"] = calc_bollinger(close)
    ind["stoch_k"], ind["stoch_d"] = calc_stoch_rsi(close)
    ind["williams_r"] = calc_williams_r(df)
    ind["cci"] = calc_cci(df)
    ind["obv"] = calc_obv(df)
    ind["vwap"] = calc_vwap(df)
    ind["tenkan"], ind["kijun"], ind["senkou_a"], ind["senkou_b"] = calc_ichimoku(df)
    ind["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    ind["avg_vol"] = df["volume"].rolling(20).mean()
    return ind

def get_signal_at(df, ind, i):
    """Считаем сигнал на индексе i, используя только данные до i (без заглядывания вперёд)"""
    if i < 60:
        return None

    close = df["close"]
    last_close = close.iloc[i]
    last_rsi   = ind["rsi"].iloc[i]
    last_hist  = ind["macd_hist"].iloc[i]
    prev_hist  = ind["macd_hist"].iloc[i-1]
    last_vol   = df["volume"].iloc[i]
    avg_vol    = ind["avg_vol"].iloc[i]
    vol_spike  = last_vol > avg_vol * 1.5 if not np.isnan(avg_vol) else False
    atr        = ind["atr"].iloc[i]

    if np.isnan(last_rsi) or np.isnan(last_hist) or np.isnan(atr):
        return None

    bullish_ema = ind["ema9"].iloc[i] > ind["ema21"].iloc[i] > ind["ema50"].iloc[i]
    bearish_ema = ind["ema9"].iloc[i] < ind["ema21"].iloc[i] < ind["ema50"].iloc[i]
    macd_bull   = prev_hist < 0 and last_hist > 0
    macd_bear   = prev_hist > 0 and last_hist < 0
    bb_break_up = last_close > ind["bb_up"].iloc[i]
    bb_break_dn = last_close < ind["bb_low"].iloc[i]

    last_stoch_k = ind["stoch_k"].iloc[i]
    last_stoch_d = ind["stoch_d"].iloc[i]
    prev_stoch_k = ind["stoch_k"].iloc[i-1]
    last_wr      = ind["williams_r"].iloc[i]
    last_cci     = ind["cci"].iloc[i]
    last_obv     = ind["obv"].iloc[i]
    prev_obv     = ind["obv"].iloc[i-5] if i >= 5 else ind["obv"].iloc[0]
    last_vwap    = ind["vwap"].iloc[i]

    stoch_bull = (not np.isnan(prev_stoch_k) and not np.isnan(last_stoch_d) and
                  prev_stoch_k <= last_stoch_d and last_stoch_k > last_stoch_d and last_stoch_k < 30)
    stoch_bear = (not np.isnan(prev_stoch_k) and not np.isnan(last_stoch_d) and
                  prev_stoch_k >= last_stoch_d and last_stoch_k < last_stoch_d and last_stoch_k > 70)

    sa, sb = ind["senkou_a"].iloc[i], ind["senkou_b"].iloc[i]
    above_cloud = (not np.isnan(sa) and not np.isnan(sb) and last_close > max(sa, sb))
    below_cloud = (not np.isnan(sa) and not np.isnan(sb) and last_close < min(sa, sb))
    tk_cross_bull = ind["tenkan"].iloc[i] > ind["kijun"].iloc[i]
    tk_cross_bear = ind["tenkan"].iloc[i] < ind["kijun"].iloc[i]

    obv_rising  = last_obv > prev_obv
    obv_falling = last_obv < prev_obv
    above_vwap  = last_close > last_vwap
    below_vwap  = last_close < last_vwap

    buy_score = sell_score = 0
    fired_buy = []
    fired_sell = []

    if last_rsi < 30:   buy_score += 3; fired_buy.append("rsi_oversold_strong")
    elif last_rsi < 40: buy_score += 2; fired_buy.append("rsi_oversold")
    elif last_rsi < 50: buy_score += 1
    if last_rsi > 70:   sell_score += 3; fired_sell.append("rsi_overbought_strong")
    elif last_rsi > 60: sell_score += 2; fired_sell.append("rsi_overbought")
    elif last_rsi > 50: sell_score += 1

    if macd_bull:  buy_score  += 3; fired_buy.append("macd_bull_cross")
    if macd_bear:  sell_score += 3; fired_sell.append("macd_bear_cross")
    if last_hist > 0: buy_score  += 1
    else:             sell_score += 1

    if bullish_ema: buy_score  += 2; fired_buy.append("ema_bullish")
    if bearish_ema: sell_score += 2; fired_sell.append("ema_bearish")

    if bb_break_up and vol_spike: buy_score  += 2; fired_buy.append("bb_break_up_vol")
    if bb_break_dn and vol_spike: sell_score += 2; fired_sell.append("bb_break_dn_vol")

    if stoch_bull: buy_score  += 2; fired_buy.append("stoch_bull_cross")
    if stoch_bear: sell_score += 2; fired_sell.append("stoch_bear_cross")
    if last_stoch_k < 20: buy_score  += 1; fired_buy.append("stoch_oversold")
    if last_stoch_k > 80: sell_score += 1; fired_sell.append("stoch_overbought")

    if last_wr < -80:  buy_score  += 2; fired_buy.append("williams_oversold")
    if last_wr > -20:  sell_score += 2; fired_sell.append("williams_overbought")

    if last_cci < -100: buy_score  += 2; fired_buy.append("cci_oversold")
    if last_cci > 100:  sell_score += 2; fired_sell.append("cci_overbought")

    if obv_rising and last_close > close.iloc[i-1]:   buy_score  += 1; fired_buy.append("obv_confirms_up")
    if obv_falling and last_close < close.iloc[i-1]:  sell_score += 1; fired_sell.append("obv_confirms_down")

    if above_vwap: buy_score  += 1; fired_buy.append("above_vwap")
    if below_vwap: sell_score += 1; fired_sell.append("below_vwap")

    if above_cloud and tk_cross_bull: buy_score  += 3; fired_buy.append("ichimoku_strong_bull")
    elif above_cloud:                 buy_score  += 1; fired_buy.append("ichimoku_above_cloud")
    if below_cloud and tk_cross_bear: sell_score += 3; fired_sell.append("ichimoku_strong_bear")
    elif below_cloud:                 sell_score += 1; fired_sell.append("ichimoku_below_cloud")

    if vol_spike:
        if last_close > close.iloc[i-1]: buy_score  += 1; fired_buy.append("vol_spike_up")
        else:                            sell_score += 1; fired_sell.append("vol_spike_down")

    if   buy_score >= 12:                                signal = "STRONG_BUY"
    elif buy_score >= 7  and buy_score > sell_score + 3: signal = "BUY"
    elif sell_score >= 12:                               signal = "STRONG_SELL"
    elif sell_score >= 7 and sell_score > buy_score + 3: signal = "SELL"
    else:                                                signal = "HOLD"

    if signal == "HOLD":
        return None

    tp1 = last_close + atr * 1.5
    tp2 = last_close + atr * 3.0
    sl  = last_close - atr * 1.0
    if "SELL" in signal:
        tp1 = last_close - atr * 1.5
        tp2 = last_close - atr * 3.0
        sl  = last_close + atr * 1.0

    return {
        "index": i, "time": df.index[i], "signal": signal,
        "price": last_close, "tp1": tp1, "tp2": tp2, "sl": sl,
        "buy_score": buy_score, "sell_score": sell_score,
        "fired": fired_buy if "BUY" in signal else fired_sell,
    }

def simulate_outcome(df, signal, max_hours=72):
    """Смотрим что случилось после сигнала: TP1, TP2, SL, или ничего за max_hours"""
    i = signal["index"]
    is_buy = "BUY" in signal["signal"]
    future = df.iloc[i+1 : i+1+max_hours]
    if len(future) == 0:
        return "no_data"

    tp1_hit = tp2_hit = sl_hit = False
    tp1_time = tp2_time = sl_time = None

    for idx, row in future.iterrows():
        if is_buy:
            if not sl_hit and row["low"] <= signal["sl"]:
                sl_hit = True; sl_time = idx
            if not tp1_hit and row["high"] >= signal["tp1"]:
                tp1_hit = True; tp1_time = idx
            if not tp2_hit and row["high"] >= signal["tp2"]:
                tp2_hit = True; tp2_time = idx
        else:
            if not sl_hit and row["high"] >= signal["sl"]:
                sl_hit = True; sl_time = idx
            if not tp1_hit and row["low"] <= signal["tp1"]:
                tp1_hit = True; tp1_time = idx
            if not tp2_hit and row["low"] <= signal["tp2"]:
                tp2_hit = True; tp2_time = idx
        if sl_hit and tp2_hit:
            break

    # Что произошло раньше — SL или TP?
    if sl_hit and (not tp1_hit or sl_time <= tp1_time):
        return "sl_first"
    elif tp2_hit:
        return "tp2_hit"
    elif tp1_hit:
        return "tp1_only"
    else:
        return "no_outcome"

# ── MAIN BACKTEST ──────────────────────────────────────────────────────────────

async def backtest_symbol(session, symbol, days=60):
    df = await fetch_klines_full(session, symbol, "1h", days)
    if df is None or len(df) < 200:
        return []

    ind = precompute(df)
    signals = []
    last_signal_idx = -999

    for i in range(60, len(df) - 1):
        # Не чаще раза в 4 часа (как антиспам в реальном боте)
        if i - last_signal_idx < 4:
            continue
        sig = get_signal_at(df, ind, i)
        if sig is None:
            continue
        outcome = simulate_outcome(df, sig)
        sig["outcome"] = outcome
        sig["symbol"] = symbol
        signals.append(sig)
        last_signal_idx = i

    return signals

async def main():
    print(f"🔄 Backtest начат: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"📊 Пары: {len(WATCHLIST)}, период: 60 дней, таймфрейм: 1H\n")

    all_signals = []
    async with aiohttp.ClientSession() as session:
        for symbol in WATCHLIST:
            print(f"  Обрабатываю {symbol}...")
            sigs = await backtest_symbol(session, symbol, days=60)
            all_signals.extend(sigs)
            print(f"    → {len(sigs)} сигналов")
            await asyncio.sleep(0.3)

    print(f"\n✅ Всего сигналов за 60 дней: {len(all_signals)}\n")

    # ── ОБЩАЯ СТАТИСТИКА ──
    df_sig = pd.DataFrame(all_signals)
    if len(df_sig) == 0:
        print("Нет сигналов для анализа")
        return

    print("="*60)
    print("ОБЩАЯ СТАТИСТИКА")
    print("="*60)

    outcome_counts = df_sig["outcome"].value_counts()
    total = len(df_sig)
    wins = outcome_counts.get("tp2_hit", 0) + outcome_counts.get("tp1_only", 0)
    losses = outcome_counts.get("sl_first", 0)
    no_outcome = outcome_counts.get("no_outcome", 0) + outcome_counts.get("no_data", 0)

    winrate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\nВсего сигналов: {total}")
    print(f"  TP2 достигнут (полный успех): {outcome_counts.get('tp2_hit', 0)} ({outcome_counts.get('tp2_hit',0)/total*100:.1f}%)")
    print(f"  TP1 достигнут (частичный успех): {outcome_counts.get('tp1_only', 0)} ({outcome_counts.get('tp1_only',0)/total*100:.1f}%)")
    print(f"  SL сработал (убыток): {outcome_counts.get('sl_first', 0)} ({outcome_counts.get('sl_first',0)/total*100:.1f}%)")
    print(f"  Без исхода за 72ч: {no_outcome} ({no_outcome/total*100:.1f}%)")
    print(f"\n🎯 ВИНРЕЙТ (TP vs SL): {winrate:.1f}%")

    # По типу сигнала
    print(f"\n--- По типу сигнала ---")
    for sig_type in ["STRONG_BUY", "BUY", "SELL", "STRONG_SELL"]:
        sub = df_sig[df_sig["signal"] == sig_type]
        if len(sub) == 0: continue
        sub_wins = len(sub[sub["outcome"].isin(["tp2_hit","tp1_only"])])
        sub_losses = len(sub[sub["outcome"] == "sl_first"])
        sub_wr = sub_wins/(sub_wins+sub_losses)*100 if (sub_wins+sub_losses)>0 else 0
        print(f"  {sig_type}: {len(sub)} сигналов, винрейт {sub_wr:.1f}%")

    # По парам
    print(f"\n--- По парам (топ-5 по винрейту, мин. 5 сигналов) ---")
    pair_stats = []
    for symbol in WATCHLIST:
        sub = df_sig[df_sig["symbol"] == symbol]
        if len(sub) < 5: continue
        sub_wins = len(sub[sub["outcome"].isin(["tp2_hit","tp1_only"])])
        sub_losses = len(sub[sub["outcome"] == "sl_first"])
        sub_wr = sub_wins/(sub_wins+sub_losses)*100 if (sub_wins+sub_losses)>0 else 0
        pair_stats.append((symbol, len(sub), sub_wr))
    pair_stats.sort(key=lambda x: x[2], reverse=True)
    for symbol, count, wr in pair_stats[:5]:
        print(f"  {symbol}: {count} сигналов, винрейт {wr:.1f}%")

    print(f"\n--- Худшие пары ---")
    for symbol, count, wr in pair_stats[-5:]:
        print(f"  {symbol}: {count} сигналов, винрейт {wr:.1f}%")

    # ── АНАЛИЗ ИНДИКАТОРОВ ──
    print(f"\n{'='*60}")
    print("КАКИЕ ИНДИКАТОРЫ РЕАЛЬНО РАБОТАЮТ")
    print(f"{'='*60}\n")

    indicator_stats = {}
    for _, row in df_sig.iterrows():
        is_win = row["outcome"] in ["tp2_hit", "tp1_only"]
        is_loss = row["outcome"] == "sl_first"
        if not (is_win or is_loss):
            continue
        for ind_name in row["fired"]:
            if ind_name not in indicator_stats:
                indicator_stats[ind_name] = {"wins": 0, "losses": 0}
            if is_win:
                indicator_stats[ind_name]["wins"] += 1
            else:
                indicator_stats[ind_name]["losses"] += 1

    ind_results = []
    for name, stats in indicator_stats.items():
        total_ind = stats["wins"] + stats["losses"]
        if total_ind < 5: continue
        wr = stats["wins"] / total_ind * 100
        ind_results.append((name, total_ind, wr))
    ind_results.sort(key=lambda x: x[2], reverse=True)

    print("Индикатор → винрейт когда сработал (мин. 5 раз):\n")
    for name, count, wr in ind_results:
        bar = "█" * int(wr/5)
        print(f"  {name:<25} {wr:>5.1f}%  ({count:>3} раз)  {bar}")

    # Сохраняем детальные результаты в JSON
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_signals": total,
        "winrate": round(winrate, 1),
        "by_signal_type": {
            sig_type: {
                "count": len(df_sig[df_sig["signal"]==sig_type]),
            } for sig_type in ["STRONG_BUY","BUY","SELL","STRONG_SELL"]
        },
        "indicator_performance": [
            {"indicator": name, "count": count, "winrate": round(wr,1)}
            for name, count, wr in ind_results
        ],
        "pair_performance": [
            {"symbol": s, "count": c, "winrate": round(wr,1)}
            for s, c, wr in pair_stats
        ],
    }
    with open("backtest_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Детальные результаты сохранены в backtest_results.json")

if __name__ == "__main__":
    asyncio.run(main())
