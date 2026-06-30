"""
ML Signal Model v2 — финальная попытка
=========================================
Изменения от первой попытки:
1. Таймфрейм 4H вместо 1H (меньше шума)
2. Цель: бинарная классификация "цена через 24ч выше на >=1%?" вместо TP/SL гонки
3. Добавлены межрыночные признаки (корреляция/бета к BTC)
4. Walk-forward валидация на нескольких окнах вместо одного split
5. L2-регуляризация, меньше глубина деревьев (против переобучения)

Запуск:
    pip3 install scikit-learn pandas numpy aiohttp --break-system-packages
    python3 ml_v2.py
"""

import asyncio
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import aiohttp

BINANCE_BASE = "https://api.binance.com/api/v3"

WATCHLIST = [
    "ETHUSDT","SOLUSDT","BNBUSDT","ENAUSDT",
    "AVAXUSDT","DOTUSDT","LINKUSDT","ARBUSDT","OPUSDT",
    "INJUSDT","SUIUSDT","TIAUSDT","WIFUSDT","PENDLEUSDT",
]  # BTC исключён из watchlist — используется как референс-признак

# ── DATA FETCH ────────────────────────────────────────────────────────────────

async def fetch_klines_full(session, symbol, interval="4h", days=180):
    limit = 1000
    all_data = []
    end_time = None
    needed = days * 6 + 200  # 6 свечей 4H в сутках

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

# ── INDICATORS ────────────────────────────────────────────────────────────────

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

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

def calc_adx(df, period=14):
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

def calc_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()

def build_features(df, btc_df):
    """Признаки включая межрыночные (относительно BTC)"""
    close = df["close"]
    f = pd.DataFrame(index=df.index)

    rsi = calc_rsi(close)
    f["rsi_dist_50"] = rsi - 50

    macd, macd_sig, macd_hist = calc_macd(close)
    f["macd_hist_norm"] = macd_hist / close
    f["macd_hist_change"] = macd_hist.diff()

    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    f["ema9_21_gap"]  = (ema9 - ema21) / close
    f["ema21_50_gap"] = (ema21 - ema50) / close
    f["price_vs_ema50"] = (close - ema50) / close

    bb_up, bb_mid, bb_low = calc_bollinger(close)
    f["bb_width"] = (bb_up - bb_low) / bb_mid
    f["bb_position"] = (close - bb_low) / (bb_up - bb_low + 1e-10)

    adx, plus_di, minus_di = calc_adx(df)
    f["adx"] = adx
    f["di_diff"] = plus_di - minus_di

    avg_vol = df["volume"].rolling(20).mean()
    f["vol_ratio"] = df["volume"] / (avg_vol + 1e-10)

    atr = (df["high"] - df["low"]).rolling(14).mean()
    f["atr_norm"] = atr / close

    f["ret_1"]  = close.pct_change(1)
    f["ret_3"]  = close.pct_change(3)
    f["ret_6"]  = close.pct_change(6)
    f["ret_12"] = close.pct_change(12)

    high_24 = df["high"].rolling(12).max()  # ~48ч на 4H ТФ
    low_24  = df["low"].rolling(12).min()
    f["range_position"] = (close - low_24) / (high_24 - low_24 + 1e-10)

    vwap = calc_vwap(df)
    f["price_vs_vwap"] = (close - vwap) / close

    # ── Межрыночные признаки относительно BTC ──
    btc_aligned = btc_df["close"].reindex(df.index, method="ffill")
    btc_ret_6  = btc_aligned.pct_change(6)
    btc_ret_12 = btc_aligned.pct_change(12)
    own_ret_6  = close.pct_change(6)
    own_ret_12 = close.pct_change(12)

    f["btc_ret_6"]  = btc_ret_6
    f["btc_ret_12"] = btc_ret_12
    f["relative_strength_vs_btc"] = own_ret_12 - btc_ret_12  # опережает/отстаёт от BTC

    # Rolling корреляция с BTC за последние 30 свечей
    f["corr_with_btc_30"] = close.pct_change().rolling(30).corr(btc_aligned.pct_change())

    f["close"] = close
    return f

FEATURE_COLS = [
    "rsi_dist_50", "macd_hist_norm", "macd_hist_change",
    "ema9_21_gap", "ema21_50_gap", "price_vs_ema50",
    "bb_width", "bb_position", "adx", "di_diff",
    "vol_ratio", "atr_norm", "ret_1", "ret_3", "ret_6", "ret_12",
    "range_position", "price_vs_vwap",
    "btc_ret_6", "btc_ret_12", "relative_strength_vs_btc", "corr_with_btc_30",
]

# ── LABELS: упрощённая цель ──────────────────────────────────────────────────

def make_simple_labels(df, horizon=6, threshold=0.01):
    """
    horizon=6 свечей по 4H = 24 часа.
    Label = 1 если цена выросла на >= threshold (1%) за этот период, иначе 0.
    Игнорируем боковик (между -1% и +1%) — это NaN, чтобы не путать модель шумом.
    """
    close = df["close"]
    future_ret = close.shift(-horizon) / close - 1
    labels = pd.Series(np.nan, index=df.index)
    labels[future_ret >= threshold]  = 1
    labels[future_ret <= -threshold] = 0
    return labels

# ── MAIN PIPELINE ────────────────────────────────────────────────────────────────

async def collect_dataset(days=180):
    async with aiohttp.ClientSession() as session:
        print("  Скачиваю BTCUSDT (референс)...")
        btc_df = await fetch_klines_full(session, "BTCUSDT", "4h", days)
        if btc_df is None:
            raise RuntimeError("Не удалось скачать BTC данные")

        all_rows = []
        for symbol in WATCHLIST:
            print(f"  Скачиваю {symbol}...")
            df = await fetch_klines_full(session, symbol, "4h", days)
            if df is None or len(df) < 100:
                print(f"    пропуск (мало данных)")
                continue

            feats = build_features(df, btc_df)
            feats["label"] = make_simple_labels(df)
            feats["symbol"] = symbol
            all_rows.append(feats)
            await asyncio.sleep(0.3)

    full = pd.concat(all_rows)
    full = full.reset_index().rename(columns={"index": "time"})
    return full

def walk_forward_validate(df):
    """
    Вместо одного train/test split делаем 3 окна walk-forward:
    каждый раз учим на всём что было раньше, тестируем на следующем куске.
    Это даёт более честную оценку чем единственный split.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    df = df.dropna(subset=FEATURE_COLS + ["label"]).copy()
    df = df.sort_values("time").reset_index(drop=True)
    print(f"\nВсего строк с полными данными: {len(df)}")
    print(f"Баланс классов: {df['label'].mean()*100:.1f}% положительных (рост >=1%)")

    n = len(df)
    # 4 окна: учим на первых 40/55/70%, тестируем на следующих 15%
    splits = [(0.40, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.00)]

    all_aucs = []
    all_test_proba = []
    all_test_labels = []

    for i, (train_end_pct, test_end_pct) in enumerate(splits):
        train_end = int(n * train_end_pct)
        test_end  = int(n * test_end_pct)

        train = df.iloc[:train_end]
        test  = df.iloc[train_end:test_end]
        if len(test) < 30:
            continue

        X_train, y_train = train[FEATURE_COLS], train["label"]
        X_test,  y_test  = test[FEATURE_COLS],  test["label"]

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.03,
            subsample=0.7, random_state=42,
            min_samples_leaf=20,  # против переобучения на мелких листьях
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        all_aucs.append(auc)
        all_test_proba.extend(proba)
        all_test_labels.extend(y_test.values)

        print(f"\nОкно {i+1}: train={len(train)}, test={len(test)}, AUC={auc:.3f}")

    print(f"\n{'='*60}")
    print(f"СРЕДНИЙ AUC по всем окнам: {np.mean(all_aucs):.3f} (std={np.std(all_aucs):.3f})")
    print(f"{'='*60}")

    # Финальная модель на всех данных кроме последнего held-out куска — для feature importance
    final_train = df.iloc[:int(n*0.85)]
    final_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.03,
        subsample=0.7, random_state=42, min_samples_leaf=20,
    )
    final_model.fit(final_train[FEATURE_COLS], final_train["label"])
    importances = sorted(zip(FEATURE_COLS, final_model.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print(f"\nТоп-10 важных признаков (по финальной модели):")
    for name, imp in importances[:10]:
        bar = "█" * int(imp * 100)
        print(f"  {name:<28} {imp:.3f}  {bar}")

    # Винрейт по порогам на объединённом held-out наборе
    all_test_proba = np.array(all_test_proba)
    all_test_labels = np.array(all_test_labels)
    print(f"\nВинрейт (доля верных предсказаний) по порогам уверенности:")
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        mask_up = all_test_proba >= threshold
        mask_down = all_test_proba <= (1 - threshold)
        n_signals = mask_up.sum() + mask_down.sum()
        if n_signals < 15:
            print(f"  Порог {threshold}: {n_signals} сигналов — недостаточно")
            continue
        correct = (all_test_labels[mask_up] == 1).sum() + (all_test_labels[mask_down] == 0).sum()
        acc = correct / n_signals * 100
        print(f"  Порог >={threshold}: {n_signals} сигналов, точность {acc:.1f}%")

    results = {
        "mean_auc": round(float(np.mean(all_aucs)), 3),
        "std_auc": round(float(np.std(all_aucs)), 3),
        "auc_per_window": [round(float(a), 3) for a in all_aucs],
        "feature_importance": [{"feature": n, "importance": round(float(i),4)} for n,i in importances],
        "class_balance_pct_positive": round(float(df['label'].mean()*100), 1),
    }
    with open("ml_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Результаты сохранены в ml_v2_results.json")

async def main():
    print(f"🔄 ML v2 Pipeline начат: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"📊 Таймфрейм 4H, 180 дней, {len(WATCHLIST)} пар + BTC референс\n")

    full_df = await collect_dataset(days=180)
    print(f"\n✅ Собрано {len(full_df)} строк данных\n")

    walk_forward_validate(full_df)

if __name__ == "__main__":
    asyncio.run(main())
