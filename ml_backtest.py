"""
ML Signal Model для Crypto Bot
================================
Вместо ручных весов индикаторов — обучаем модель (Gradient Boosting)
предсказывать вероятность что TP будет достигнут раньше SL.

Важно: используем time-based split (не random!), чтобы не было утечки
данных из будущего. Учим на первых ~75% времени, проверяем на последних ~25%.

Запуск:
    python3 ml_backtest.py

Требует: pip install scikit-learn pandas numpy aiohttp lightgbm --break-system-packages
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

async def fetch_klines_full(session, symbol, interval="1h", days=90):
    limit = 1000
    all_data = []
    end_time = None
    needed = days * 24 + 200

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

# ── FEATURE ENGINEERING ─────────────────────────────────────────────────────────

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

def calc_adx(df, period=14):
    """Average Directional Index — сила тренда, новый признак не использовавшийся раньше"""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period).mean()
    return adx, plus_di, minus_di

def build_features(df):
    """Строим набор признаков для ML. Каждый признак — число, без if-логики."""
    close = df["close"]
    f = pd.DataFrame(index=df.index)

    rsi = calc_rsi(close)
    f["rsi"] = rsi
    f["rsi_dist_50"] = rsi - 50  # расстояние от нейтрали, со знаком

    macd, macd_sig, macd_hist = calc_macd(close)
    f["macd_hist"] = macd_hist
    f["macd_hist_norm"] = macd_hist / close  # нормализуем на цену (сравнимо между парами)
    f["macd_hist_change"] = macd_hist.diff()

    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    f["ema9_21_gap"]  = (ema9 - ema21) / close
    f["ema21_50_gap"] = (ema21 - ema50) / close
    f["price_vs_ema50"] = (close - ema50) / close

    bb_up, bb_mid, bb_low = calc_bollinger(close)
    bb_width = (bb_up - bb_low) / bb_mid
    f["bb_width"] = bb_width
    f["bb_position"] = (close - bb_low) / (bb_up - bb_low + 1e-10)  # 0=нижняя, 1=верхняя

    stoch_k, stoch_d = calc_stoch_rsi(close)
    f["stoch_k"] = stoch_k
    f["stoch_k_d_gap"] = stoch_k - stoch_d

    f["williams_r"] = calc_williams_r(df)
    f["cci"] = calc_cci(df)

    obv = calc_obv(df)
    f["obv_change_5"] = obv.diff(5) / (df["volume"].rolling(20).sum() + 1)  # нормализуем

    vwap = calc_vwap(df)
    f["price_vs_vwap"] = (close - vwap) / close

    adx, plus_di, minus_di = calc_adx(df)
    f["adx"] = adx
    f["di_diff"] = plus_di - minus_di

    # Объём
    avg_vol = df["volume"].rolling(20).mean()
    f["vol_ratio"] = df["volume"] / (avg_vol + 1e-10)

    # Волатильность
    atr = (df["high"] - df["low"]).rolling(14).mean()
    f["atr_norm"] = atr / close
    f["atr"] = atr  # сырое значение, нужно для TP/SL расчёта

    # Momentum на разных горизонтах
    f["ret_3h"]  = close.pct_change(3)
    f["ret_6h"]  = close.pct_change(6)
    f["ret_12h"] = close.pct_change(12)
    f["ret_24h"] = close.pct_change(24)

    # Положение цены в недавнем диапазоне
    high_24 = df["high"].rolling(24).max()
    low_24  = df["low"].rolling(24).min()
    f["range_position_24h"] = (close - low_24) / (high_24 - low_24 + 1e-10)

    f["close"] = close  # для расчёта исходов, не используется как признак напрямую
    return f

FEATURE_COLS = [
    "rsi", "rsi_dist_50", "macd_hist_norm", "macd_hist_change",
    "ema9_21_gap", "ema21_50_gap", "price_vs_ema50",
    "bb_width", "bb_position", "stoch_k", "stoch_k_d_gap",
    "williams_r", "cci", "obv_change_5", "price_vs_vwap",
    "adx", "di_diff", "vol_ratio", "atr_norm",
    "ret_3h", "ret_6h", "ret_12h", "ret_24h", "range_position_24h",
]

# ── LABEL GENERATION (что произошло после каждой точки) ────────────────────────

def make_labels(df, direction="long", max_hours=72, tp_atr=1.5, sl_atr=1.0):
    """
    Для каждой точки i: если бы мы открыли позицию (long/short) с TP=1.5*ATR,
    SL=1.0*ATR — что случилось бы раньше? Возвращает 1 (TP first), 0 (SL first), NaN (no outcome).
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    atr   = (df["high"] - df["low"]).rolling(14).mean().values
    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - 1):
        if np.isnan(atr[i]):
            continue
        price = close[i]
        if direction == "long":
            tp = price + atr[i] * tp_atr
            sl = price - atr[i] * sl_atr
        else:
            tp = price - atr[i] * tp_atr
            sl = price + atr[i] * sl_atr

        end = min(i + 1 + max_hours, n)
        outcome = np.nan
        for j in range(i + 1, end):
            if direction == "long":
                hit_sl = low[j] <= sl
                hit_tp = high[j] >= tp
            else:
                hit_sl = high[j] >= sl
                hit_tp = low[j] <= tp
            if hit_sl and hit_tp:
                # В одной свече и то и то — консервативно считаем SL первым (хуже для нас)
                outcome = 0
                break
            elif hit_sl:
                outcome = 0
                break
            elif hit_tp:
                outcome = 1
                break
        labels[i] = outcome
    return labels

# ── MAIN PIPELINE ────────────────────────────────────────────────────────────────

async def collect_dataset(days=90):
    """Собираем фичи + лейблы по всем парам, помечаем символом и временем"""
    all_rows = []
    async with aiohttp.ClientSession() as session:
        for symbol in WATCHLIST:
            print(f"  Скачиваю {symbol}...")
            df = await fetch_klines_full(session, symbol, "1h", days)
            if df is None or len(df) < 200:
                print(f"    пропуск (мало данных)")
                continue

            feats = build_features(df)
            labels_long  = make_labels(df, "long")
            labels_short = make_labels(df, "short")

            feats["symbol"] = symbol
            feats["label_long"]  = labels_long
            feats["label_short"] = labels_short
            all_rows.append(feats)
            await asyncio.sleep(0.3)

    full = pd.concat(all_rows)
    full = full.reset_index().rename(columns={"index": "time"})
    return full

def train_and_evaluate(full_df):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    # Берём только строки где есть полный набор фичей и лейбл
    df = full_df.dropna(subset=FEATURE_COLS + ["label_long", "label_short"]).copy()
    df = df.sort_values("time")

    print(f"\nВсего строк с полными данными: {len(df)}")

    # Time-based split: 75% на обучение, 25% на тест (по времени, без перемешивания!)
    split_time = df["time"].quantile(0.75)
    train = df[df["time"] <= split_time]
    test  = df[df["time"] >  split_time]
    print(f"Train: {len(train)} строк (до {split_time})")
    print(f"Test:  {len(test)} строк (после {split_time})")

    results = {}
    for direction, label_col in [("LONG", "label_long"), ("SHORT", "label_short")]:
        print(f"\n{'='*60}")
        print(f"Модель для {direction}")
        print(f"{'='*60}")

        X_train = train[FEATURE_COLS]
        y_train = train[label_col]
        X_test  = test[FEATURE_COLS]
        y_test  = test[label_col]

        # Базовый винрейт (что было бы если торговать каждый сигнал без фильтра)
        baseline_wr = y_test.mean() * 100
        print(f"Базовый винрейт без фильтра (торговать всё): {baseline_wr:.1f}%")

        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC на тесте: {auc:.3f}  (0.5 = случайность, 1.0 = идеально)")

        # Смотрим винрейт если торговать только сигналы где модель уверена (proba > 0.6)
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
            mask = proba >= threshold
            n_signals = mask.sum()
            if n_signals < 10:
                print(f"  Порог {threshold}: только {n_signals} сигналов — недостаточно данных")
                continue
            wr = y_test[mask].mean() * 100
            print(f"  Порог уверенности >={threshold}: {n_signals} сигналов, винрейт {wr:.1f}%")

        # Важность признаков
        importances = sorted(zip(FEATURE_COLS, model.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        print(f"\nТоп-10 важных признаков:")
        for name, imp in importances[:10]:
            bar = "█" * int(imp * 100)
            print(f"  {name:<22} {imp:.3f}  {bar}")

        results[direction] = {
            "baseline_winrate": round(baseline_wr, 1),
            "auc": round(auc, 3),
            "feature_importance": [{"feature": n, "importance": round(i,4)} for n,i in importances],
        }

    with open("ml_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Результаты сохранены в ml_results.json")

async def main():
    print(f"🔄 ML Pipeline начат: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"📊 Собираю данные за 90 дней по {len(WATCHLIST)} парам...\n")

    full_df = await collect_dataset(days=90)
    print(f"\n✅ Собрано {len(full_df)} строк данных\n")

    train_and_evaluate(full_df)

if __name__ == "__main__":
    asyncio.run(main())
