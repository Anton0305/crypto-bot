"""
Crypto Signals Bot v4.0
Новое:
- 13 индикаторов (добавлены Stochastic RSI, Williams %R, CCI, OBV, Ichimoku, VWAP)
- AI Sentiment Analysis новостей через Claude API
- Новостной сентимент влияет на итоговый сигнал
- Улучшенная система скоринга
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import aiohttp
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import io

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHANNEL_ID       = os.getenv("CHANNEL_ID", "@CryptoSignalsBotPro")
NEWSAPI_TOKEN    = os.getenv("NEWSAPI_TOKEN", "YOUR_NEWSAPI_TOKEN")

WATCHLIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","ENAUSDT",
    "AVAXUSDT","DOTUSDT","LINKUSDT","ARBUSDT","OPUSDT",
    "INJUSDT","SUIUSDT","TIAUSDT","WIFUSDT","PENDLEUSDT",
]

PRICE_LEVELS = {
    "BTCUSDT":  [50000,55000,60000,65000,70000,75000,100000],
    "ETHUSDT":  [2000,2500,3000,3500,4000],
    "SOLUSDT":  [50,100,150,200,250],
    "BNBUSDT":  [400,500,600,700,800],
}

CHECK_INTERVAL     = 300
NEWS_INTERVAL      = 7200
DIGEST_INTERVAL    = 86400
EDUCATION_INTERVAL = 14400
BTC_ANALYSIS_INTERVAL = 86400
SCREENER_INTERVAL  = 21600

BINANCE_BASE   = "https://api.binance.com/api/v3"
FEAR_GREED_API = "https://api.alternative.me/fng/?limit=1"
NEWSAPI_BASE   = "https://newsapi.org/v2/everything"

last_alerts: dict = {}
last_price_alerts: dict = {}
sentiment_cache: dict = {"score": 0, "summary": "", "ts": 0}

# ── BINANCE ───────────────────────────────────────────────────────────────────

async def fetch_klines(session, symbol, interval="1h", limit=150):
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
        data = await r.json()
    if not isinstance(data, list) or len(data) < 30:
        return None
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df.set_index("time", inplace=True)
    return df

async def fetch_ticker(session, symbol):
    url = f"{BINANCE_BASE}/ticker/24hr"
    async with session.get(url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)) as r:
        return await r.json()

async def fetch_all_tickers(session):
    url = f"{BINANCE_BASE}/ticker/24hr"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
        return await r.json()

async def fetch_fear_greed(session):
    try:
        async with session.get(FEAR_GREED_API, timeout=aiohttp.ClientTimeout(total=8)) as r:
            data = await r.json()
            return int(data["data"][0]["value"]), data["data"][0]["value_classification"]
    except Exception:
        return None, None

async def fetch_btc_dominance(session):
    try:
        async with session.get("https://api.coingecko.com/api/v3/global",
                               timeout=aiohttp.ClientTimeout(total=8)) as r:
            data = await r.json()
            return round(data["data"]["market_cap_percentage"]["btc"], 1)
    except Exception:
        return None

# ── SENTIMENT ANALYSIS (бесплатный, на ключевых словах) ───────────────────────

# Веса слов для оценки сентимента. Подобраны по силе влияния на крипторынок.
BULLISH_WORDS = {
    "etf approved": 4, "etf approval": 4, "spot etf": 3,
    "rate cut": 3, "fed cuts": 3, "interest rate cut": 3,
    "all-time high": 3, "ath": 3, "record high": 3,
    "institutional adoption": 3, "institutional buying": 3,
    "bullish": 2, "rally": 2, "surge": 2, "soars": 2, "breakout": 2,
    "inflation falls": 2, "inflation cools": 2, "cpi lower": 2,
    "accumulation": 2, "whale buying": 2, "inflow": 2,
    "halving": 2, "upgrade": 1, "partnership": 1, "adoption": 1,
    "buy the dip": 1, "outperform": 1, "green": 1,
}

BEARISH_WORDS = {
    "sec sues": 4, "sec lawsuit": 4, "banned": 4, "ban crypto": 4,
    "hack": 3, "hacked": 3, "exploit": 3, "stolen": 3,
    "rate hike": 3, "fed hikes": 3, "interest rate increase": 3,
    "crash": 3, "plunge": 3, "collapse": 3, "sell-off": 3, "selloff": 3,
    "inflation rises": 3, "inflation surges": 3, "cpi higher": 3,
    "recession": 3, "bearish": 2, "liquidation": 2, "liquidated": 2,
    "outflow": 2, "fraud": 2, "scam": 2, "investigation": 2,
    "regulatory crackdown": 3, "crackdown": 2, "delisted": 2,
    "bankruptcy": 3, "insolvent": 3, "red": 1, "decline": 1, "drop": 1,
}

def keyword_sentiment(title: str) -> int:
    """Считаем сентимент-скор для одного заголовка по ключевым словам"""
    title_lower = title.lower()
    score = 0
    for phrase, weight in BULLISH_WORDS.items():
        if phrase in title_lower:
            score += weight
    for phrase, weight in BEARISH_WORDS.items():
        if phrase in title_lower:
            score -= weight
    return score

async def analyze_sentiment_ai(session, news_titles: list[str]) -> dict:
    """Бесплатный анализ сентимента на основе ключевых слов"""
    if not news_titles:
        return {"score": 0, "summary": "", "impact": "neutral"}

    now_ts = datetime.now(timezone.utc).timestamp()
    if now_ts - sentiment_cache["ts"] < 7200 and sentiment_cache["ts"] > 0:
        return sentiment_cache

    raw_scores = [keyword_sentiment(t) for t in news_titles if t]
    raw_scores = [s for s in raw_scores if s != 0]

    if not raw_scores:
        result = {"score": 0, "summary": "Нет значимых новостей, влияющих на рынок",
                  "impact": "neutral", "key_factor": "", "ts": now_ts}
        sentiment_cache.update(result)
        return result

    avg = sum(raw_scores) / len(raw_scores)
    # Нормализуем в диапазон -5..+5
    score = max(-5, min(5, round(avg)))

    if score >= 3:
        impact = "bullish"
        summary = "Новостной фон преимущественно позитивный для крипторынка"
    elif score >= 1:
        impact = "bullish"
        summary = "Новостной фон умеренно позитивный"
    elif score <= -3:
        impact = "bearish"
        summary = "Новостной фон преимущественно негативный для крипторынка"
    elif score <= -1:
        impact = "bearish"
        summary = "Новостной фон умеренно негативный"
    else:
        impact = "neutral"
        summary = "Новостной фон нейтральный"

    result = {"score": score, "summary": summary, "impact": impact,
              "key_factor": "", "ts": now_ts}
    sentiment_cache.update(result)
    logger.info(f"Sentiment (keyword-based): score={score}, impact={impact}")
    return result

# ── NEWS ──────────────────────────────────────────────────────────────────────

CRYPTO_KEYWORDS = "bitcoin OR ethereum OR crypto OR blockchain"

async def fetch_world_news(session):
    if NEWSAPI_TOKEN == "YOUR_NEWSAPI_TOKEN":
        return []
    try:
        params = {
            "q": CRYPTO_KEYWORDS,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": NEWSAPI_TOKEN,
        }
        async with session.get(NEWSAPI_BASE, params=params,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            return data.get("articles", [])[:8]
    except Exception as e:
        logger.error(f"NewsAPI: {e}")
        return []

def categorize_news(title: str) -> tuple:
    title_lower = title.lower()
    if any(w in title_lower for w in ["etf approved","bullish","rally","ath","adoption","rate cut","inflation falls"]):
        return "🟢", "Позитив"
    if any(w in title_lower for w in ["ban","crackdown","hack","crash","rate hike","recession","bear"]):
        return "🔴", "Негатив"
    if any(w in title_lower for w in ["federal reserve","fed","cpi","gdp","inflation","interest rate"]):
        return "🔵", "Макро"
    return "⚪", "Нейтрально"

async def send_news_digest(bot, session):
    articles = await fetch_world_news(session)
    fg_val, _ = await fetch_fear_greed(session)

    # AI анализ
    titles = [a.get("title","") for a in articles if a.get("title")]
    sentiment = await analyze_sentiment_ai(session, titles)

    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    lines = ["📰 *МИРОВЫЕ НОВОСТИ И СЕНТИМЕНТ*", f"_{now}_", ""]

    if fg_val:
        fg_e = "😱" if fg_val<=20 else "😨" if fg_val<=40 else "😐" if fg_val<=60 else "😏" if fg_val<=80 else "🤑"
        lines.append(f"{fg_e} Fear & Greed: `{fg_val}`")

    # AI сентимент блок
    if sentiment.get("summary"):
        impact_emoji = "🟢" if sentiment["impact"]=="bullish" else "🔴" if sentiment["impact"]=="bearish" else "⚪"
        score = sentiment["score"]
        score_bar = "🟢" * max(0, score) + "🔴" * max(0, -score)
        lines += [
            "",
            f"🧠 *AI Анализ сентимента:*",
            f"  {impact_emoji} {sentiment['summary']}",
            f"  Скор: `{score:+d}/5`  {score_bar}",
        ]
        if sentiment.get("key_factor"):
            lines.append(f"  Ключевой фактор: _{sentiment['key_factor']}_")
        lines.append("")

    if not articles:
        lines.append("_Новости временно недоступны_")
    else:
        for art in articles[:6]:
            title  = art.get("title","")[:100]
            source = art.get("source",{}).get("name","")
            url    = art.get("url","")
            emoji, impact = categorize_news(title)
            lines.append(f"{emoji} [{title}]({url})")
            if source: lines.append(f"_📌 {source}_")
            lines.append("")

    lines.append("_Учитывай новостной фон при торговле_")
    await bot.send_message(CHANNEL_ID, "\n".join(lines),
                           parse_mode=ParseMode.MARKDOWN,
                           disable_web_page_preview=True)

# ── INDICATORS ────────────────────────────────────────────────────────────────

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

def calc_support_resistance(df, window=20):
    highs = df["high"].rolling(window, center=True).max()
    lows  = df["low"].rolling(window, center=True).min()
    resistance = df["high"][df["high"] == highs].dropna().tail(3).values.tolist()
    support    = df["low"][df["low"] == lows].dropna().tail(3).values.tolist()
    return sorted(set(support)), sorted(set(resistance))

def analyze(df1h, df4h=None, sentiment_score=0):
    close = df1h["close"]
    vol   = df1h["volume"]

    # Базовые индикаторы
    rsi   = calc_rsi(close)
    macd, macd_sig, macd_hist = calc_macd(close)
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    bb_up, bb_mid, bb_low = calc_bollinger(close)

    # Новые индикаторы
    stoch_k, stoch_d = calc_stoch_rsi(close)
    williams_r       = calc_williams_r(df1h)
    cci              = calc_cci(df1h)
    obv              = calc_obv(df1h)
    vwap             = calc_vwap(df1h)
    tenkan, kijun, senkou_a, senkou_b = calc_ichimoku(df1h)
    support, resistance = calc_support_resistance(df1h)

    last_rsi      = rsi.iloc[-1]
    last_hist     = macd_hist.iloc[-1]
    prev_hist     = macd_hist.iloc[-2]
    last_close    = close.iloc[-1]
    last_vol      = vol.iloc[-1]
    avg_vol       = vol.rolling(20).mean().iloc[-1]
    vol_spike     = last_vol > avg_vol * 1.5
    atr           = (df1h["high"] - df1h["low"]).rolling(14).mean().iloc[-1]

    last_stoch_k  = stoch_k.iloc[-1]
    last_stoch_d  = stoch_d.iloc[-1]
    prev_stoch_k  = stoch_k.iloc[-2]
    last_wr       = williams_r.iloc[-1]
    last_cci      = cci.iloc[-1]
    last_obv      = obv.iloc[-1]
    prev_obv      = obv.iloc[-5]
    last_vwap     = vwap.iloc[-1]
    last_tenkan   = tenkan.iloc[-1]
    last_kijun    = kijun.iloc[-1]

    bullish_ema  = ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]
    bearish_ema  = ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]
    macd_bull    = prev_hist < 0 and last_hist > 0
    macd_bear    = prev_hist > 0 and last_hist < 0
    bb_break_up  = last_close > bb_up.iloc[-1]
    bb_break_dn  = last_close < bb_low.iloc[-1]

    # Stochastic RSI кросс
    stoch_bull   = prev_stoch_k <= last_stoch_d and last_stoch_k > last_stoch_d and last_stoch_k < 30
    stoch_bear   = prev_stoch_k >= last_stoch_d and last_stoch_k < last_stoch_d and last_stoch_k > 70

    # Ichimoku
    above_cloud  = last_close > max(senkou_a.iloc[-1] or 0, senkou_b.iloc[-1] or 0)
    below_cloud  = last_close < min(senkou_a.iloc[-1] or 0, senkou_b.iloc[-1] or 0)
    tk_cross_bull = last_tenkan > last_kijun
    tk_cross_bear = last_tenkan < last_kijun

    # OBV тренд
    obv_rising   = last_obv > prev_obv
    obv_falling  = last_obv < prev_obv

    # VWAP позиция
    above_vwap   = last_close > last_vwap
    below_vwap   = last_close < last_vwap

    # 4H подтверждение
    confirm_4h = 0
    if df4h is not None:
        try:
            rsi_4h   = calc_rsi(df4h["close"]).iloc[-1]
            ema9_4h  = df4h["close"].ewm(span=9,  adjust=False).mean().iloc[-1]
            ema21_4h = df4h["close"].ewm(span=21, adjust=False).mean().iloc[-1]
            macd_4h, _, hist_4h = calc_macd(df4h["close"])
            if rsi_4h < 45 and ema9_4h > ema21_4h and hist_4h.iloc[-1] > 0: confirm_4h = +3
            elif rsi_4h < 50 and ema9_4h > ema21_4h:                         confirm_4h = +1
            if rsi_4h > 55 and ema9_4h < ema21_4h and hist_4h.iloc[-1] < 0: confirm_4h = -3
            elif rsi_4h > 50 and ema9_4h < ema21_4h:                         confirm_4h = -1
        except Exception:
            pass

    buy_score = sell_score = 0
    reasons_buy = []
    reasons_sell = []

    # RSI
    if last_rsi < 30:   buy_score += 3; reasons_buy.append(f"RSI сильно перепродан ({last_rsi:.0f})")
    elif last_rsi < 40: buy_score += 2; reasons_buy.append(f"RSI перепродан ({last_rsi:.0f})")
    elif last_rsi < 50: buy_score += 1
    if last_rsi > 70:   sell_score += 3; reasons_sell.append(f"RSI сильно перекуплен ({last_rsi:.0f})")
    elif last_rsi > 60: sell_score += 2; reasons_sell.append(f"RSI перекуплен ({last_rsi:.0f})")
    elif last_rsi > 50: sell_score += 1

    # MACD
    if macd_bull:  buy_score  += 3; reasons_buy.append("MACD бычий разворот")
    if macd_bear:  sell_score += 3; reasons_sell.append("MACD медвежий разворот")
    if last_hist > 0: buy_score  += 1
    else:             sell_score += 1

    # EMA
    if bullish_ema: buy_score  += 2; reasons_buy.append("EMA бычье выравнивание")
    if bearish_ema: sell_score += 2; reasons_sell.append("EMA медвежье выравнивание")

    # Bollinger
    if bb_break_up and vol_spike: buy_score  += 2; reasons_buy.append("Пробой BB вверх + объём")
    if bb_break_dn and vol_spike: sell_score += 2; reasons_sell.append("Пробой BB вниз + объём")

    # Stochastic RSI
    if stoch_bull: buy_score  += 2; reasons_buy.append(f"Stoch RSI бычий кросс ({last_stoch_k:.0f})")
    if stoch_bear: sell_score += 2; reasons_sell.append(f"Stoch RSI медвежий кросс ({last_stoch_k:.0f})")
    if last_stoch_k < 20: buy_score  += 1; reasons_buy.append("Stoch RSI перепродан")
    if last_stoch_k > 80: sell_score += 1; reasons_sell.append("Stoch RSI перекуплен")

    # Williams %R
    if last_wr < -80:  buy_score  += 2; reasons_buy.append(f"Williams %R перепродан ({last_wr:.0f})")
    if last_wr > -20:  sell_score += 2; reasons_sell.append(f"Williams %R перекуплен ({last_wr:.0f})")

    # CCI
    if last_cci < -100:  buy_score  += 2; reasons_buy.append(f"CCI перепродан ({last_cci:.0f})")
    if last_cci > 100:   sell_score += 2; reasons_sell.append(f"CCI перекуплен ({last_cci:.0f})")

    # OBV
    if obv_rising and last_close > close.iloc[-2]:   buy_score  += 1; reasons_buy.append("OBV подтверждает рост")
    if obv_falling and last_close < close.iloc[-2]:  sell_score += 1; reasons_sell.append("OBV подтверждает падение")

    # VWAP
    if above_vwap:  buy_score  += 1; reasons_buy.append("Цена выше VWAP")
    if below_vwap:  sell_score += 1; reasons_sell.append("Цена ниже VWAP")

    # Ichimoku
    if above_cloud and tk_cross_bull:  buy_score  += 3; reasons_buy.append("Ichimoku: цена выше облака + TK бычий кросс")
    elif above_cloud:                  buy_score  += 1; reasons_buy.append("Ichimoku: цена выше облака")
    if below_cloud and tk_cross_bear:  sell_score += 3; reasons_sell.append("Ichimoku: цена ниже облака + TK медвежий кросс")
    elif below_cloud:                  sell_score += 1; reasons_sell.append("Ichimoku: цена ниже облака")

    # Объём
    if vol_spike:
        if last_close > close.iloc[-2]: buy_score  += 1; reasons_buy.append("Всплеск объёма на росте")
        else:                           sell_score += 1; reasons_sell.append("Всплеск объёма на падении")

    # 4H подтверждение
    if confirm_4h > 0:   buy_score  += confirm_4h;      reasons_buy.append(f"4H подтверждает (+{confirm_4h})")
    elif confirm_4h < 0: sell_score += abs(confirm_4h); reasons_sell.append(f"4H подтверждает (-{abs(confirm_4h)})")

    # 🧠 AI Сентимент новостей
    sentiment_adj = 0
    if sentiment_score >= 3:
        sentiment_adj = 2; reasons_buy.append(f"🧠 AI сентимент позитивный (+{sentiment_score})")
    elif sentiment_score >= 1:
        sentiment_adj = 1; reasons_buy.append(f"🧠 AI сентимент умеренно позитивный")
    elif sentiment_score <= -3:
        sentiment_adj = -2; reasons_sell.append(f"🧠 AI сентимент негативный ({sentiment_score})")
    elif sentiment_score <= -1:
        sentiment_adj = -1; reasons_sell.append(f"🧠 AI сентимент умеренно негативный")

    if sentiment_adj > 0:   buy_score  += sentiment_adj
    elif sentiment_adj < 0: sell_score += abs(sentiment_adj)

    # Итоговый сигнал
    max_score = buy_score + sell_score
    if   buy_score >= 12:                                signal = "STRONG_BUY"
    elif buy_score >= 7  and buy_score > sell_score + 3: signal = "BUY"
    elif sell_score >= 12:                               signal = "STRONG_SELL"
    elif sell_score >= 7 and sell_score > buy_score + 3: signal = "SELL"
    else:                                                signal = "HOLD"

    tp1 = last_close + atr * 1.5
    tp2 = last_close + atr * 3.0
    sl  = last_close - atr * 1.0
    if "SELL" in signal:
        tp1 = last_close - atr * 1.5
        tp2 = last_close - atr * 3.0
        sl  = last_close + atr * 1.0

    return {
        "signal": signal, "buy_score": buy_score, "sell_score": sell_score,
        "price": last_close, "rsi": last_rsi, "macd_hist": last_hist,
        "stoch_k": last_stoch_k, "stoch_d": last_stoch_d,
        "williams_r": last_wr, "cci": last_cci,
        "vwap": last_vwap, "tenkan": last_tenkan, "kijun": last_kijun,
        "ema9": ema9.iloc[-1], "ema21": ema21.iloc[-1], "ema50": ema50.iloc[-1],
        "bb_up": bb_up.iloc[-1], "bb_low": bb_low.iloc[-1],
        "vol_spike": vol_spike, "atr": atr,
        "above_cloud": above_cloud, "below_cloud": below_cloud,
        "tp1": tp1, "tp2": tp2, "sl": sl,
        "support": support, "resistance": resistance,
        "reasons_buy": reasons_buy, "reasons_sell": reasons_sell,
        "sentiment_score": sentiment_score,
        "df": df1h, "rsi_series": rsi, "macd_series": macd,
        "macd_sig_series": macd_sig, "macd_hist_series": macd_hist,
        "stoch_k_s": stoch_k, "stoch_d_s": stoch_d,
        "ema9_s": ema9, "ema21_s": ema21, "ema50_s": ema50,
        "bb_up_s": bb_up, "bb_mid_s": bb_mid, "bb_low_s": bb_low,
        "vwap_s": vwap,
    }

# ── CHART ─────────────────────────────────────────────────────────────────────

def build_chart(res, symbol):
    df = res["df"].copy().tail(60)
    color_map = {
        "STRONG_BUY":"#00ff88","BUY":"#00cc66",
        "HOLD":"#ffcc00","SELL":"#ff6644","STRONG_SELL":"#ff2200",
    }
    sig_color = color_map.get(res["signal"], "#aaa")
    fig = plt.figure(figsize=(14, 13), facecolor="#0d1117")
    gs  = GridSpec(5, 1, figure=fig, hspace=0.06, height_ratios=[3,.7,.7,.7,.7])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.spines[:].set_color("#21262d")
    x = np.arange(len(df))

    # Свечи
    for i, (_, row) in enumerate(df.iterrows()):
        c = "#26a641" if row["close"] >= row["open"] else "#f85149"
        ax1.plot([i,i],[row["low"],row["high"]], color=c, lw=0.8)
        ax1.bar(i, abs(row["close"]-row["open"]),
                bottom=min(row["open"],row["close"]), color=c, width=0.7, alpha=0.9)

    # Bollinger
    ax1.fill_between(x, res["bb_up_s"].values[-60:], res["bb_low_s"].values[-60:],
                     alpha=0.07, color="#58a6ff")
    ax1.plot(x, res["bb_up_s"].values[-60:],  color="#58a6ff", lw=0.7, alpha=0.5)
    ax1.plot(x, res["bb_mid_s"].values[-60:], color="#58a6ff", lw=0.6, alpha=0.3, ls="--")
    ax1.plot(x, res["bb_low_s"].values[-60:], color="#58a6ff", lw=0.7, alpha=0.5)

    # EMAs
    ax1.plot(x, res["ema9_s"].values[-60:],  color="#f0c419", lw=1.2, label="EMA9")
    ax1.plot(x, res["ema21_s"].values[-60:], color="#e36bdf", lw=1.2, label="EMA21")
    ax1.plot(x, res["ema50_s"].values[-60:], color="#58a6ff", lw=1.2, label="EMA50")

    # VWAP
    ax1.plot(x, res["vwap_s"].values[-60:], color="#ff9500", lw=1.0, ls="--", label="VWAP", alpha=0.8)

    # Support/Resistance
    for s in res["support"]:
        ax1.axhline(s, color="#26a641", lw=0.7, ls=":", alpha=0.6)
        ax1.text(0, s, " S", color="#26a641", fontsize=6, va="center")
    for r in res["resistance"]:
        ax1.axhline(r, color="#f85149", lw=0.7, ls=":", alpha=0.6)
        ax1.text(0, r, " R", color="#f85149", fontsize=6, va="center")

    # TP/SL
    if res["signal"] != "HOLD":
        ax1.axhline(res["tp1"], color="#26a641", lw=0.9, ls="--", alpha=0.8)
        ax1.axhline(res["tp2"], color="#26a641", lw=0.7, ls=":",  alpha=0.6)
        ax1.axhline(res["sl"],  color="#f85149", lw=0.9, ls="--", alpha=0.8)
        ax1.text(len(x)-1, res["tp1"], " TP1", color="#26a641", va="center", fontsize=7)
        ax1.text(len(x)-1, res["tp2"], " TP2", color="#26a641", va="center", fontsize=7)
        ax1.text(len(x)-1, res["sl"],  " SL",  color="#f85149", va="center", fontsize=7)

    p = res["price"]
    p_str = f"${p:,.2f}" if p >= 1 else f"${p:.4f}"
    sent_str = f" | 🧠{res['sentiment_score']:+d}" if res['sentiment_score'] != 0 else ""
    ax1.set_title(f"{symbol}  {p_str}   [{res['signal']}]{sent_str}",
                  color=sig_color, fontsize=13, fontweight="bold", pad=8, fontfamily="monospace")
    ax1.legend(fontsize=6, loc="upper left",
               facecolor="#161b22", edgecolor="#21262d", labelcolor="#8b949e")

    # Volume
    vc = ["#26a641" if df["close"].iloc[i]>=df["open"].iloc[i] else "#f85149" for i in range(len(df))]
    ax2.bar(x, df["volume"].values, color=vc, alpha=0.7, width=0.7)
    ax2.plot(x, df["volume"].rolling(20).mean().values, color="#f0c419", lw=0.8)
    ax2.set_ylabel("Vol", fontsize=6, color="#8b949e")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v>=1e6 else f"{v/1e3:.0f}K"))

    # MACD
    hist = res["macd_hist_series"].values[-60:]
    ax3.bar(x, hist, color=["#26a641" if h>=0 else "#f85149" for h in hist], alpha=0.7, width=0.7)
    ax3.plot(x, res["macd_series"].values[-60:],     color="#58a6ff", lw=1.0)
    ax3.plot(x, res["macd_sig_series"].values[-60:], color="#f0c419", lw=1.0)
    ax3.axhline(0, color="#21262d", lw=0.8)
    ax3.set_ylabel("MACD", fontsize=6, color="#8b949e")

    # RSI
    rv = res["rsi_series"].values[-60:]
    ax4.plot(x, rv, color="#e36bdf", lw=1.2, label="RSI")
    ax4.axhline(70, color="#f85149", lw=0.7, ls="--", alpha=0.6)
    ax4.axhline(30, color="#26a641", lw=0.7, ls="--", alpha=0.6)
    ax4.axhline(50, color="#21262d", lw=0.6)
    ax4.fill_between(x, rv, 70, where=(rv>=70), alpha=0.15, color="#f85149")
    ax4.fill_between(x, rv, 30, where=(rv<=30), alpha=0.15, color="#26a641")
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("RSI", fontsize=6, color="#8b949e")
    ax4.text(len(x)-1, rv[-1], f"  {rv[-1]:.0f}", color="#e36bdf", va="center", fontsize=7)

    # Stochastic RSI
    sk = res["stoch_k_s"].values[-60:]
    sd = res["stoch_d_s"].values[-60:]
    ax5.plot(x, sk, color="#00ff88", lw=1.0, label="%K")
    ax5.plot(x, sd, color="#ff9500", lw=1.0, label="%D")
    ax5.axhline(80, color="#f85149", lw=0.7, ls="--", alpha=0.6)
    ax5.axhline(20, color="#26a641", lw=0.7, ls="--", alpha=0.6)
    ax5.fill_between(x, sk, 80, where=(sk>=80), alpha=0.15, color="#f85149")
    ax5.fill_between(x, sk, 20, where=(sk<=20), alpha=0.15, color="#26a641")
    ax5.set_ylim(0, 100)
    ax5.set_ylabel("StochRSI", fontsize=6, color="#8b949e")
    ax5.legend(fontsize=6, loc="upper left",
               facecolor="#161b22", edgecolor="#21262d", labelcolor="#8b949e")

    ticks = list(range(0, len(df), 10))
    ax5.set_xticks(ticks)
    ax5.set_xticklabels([df.index[i].strftime("%d/%m %H:%M") for i in ticks],
                        rotation=30, ha="right", fontsize=7, color="#8b949e")
    for a in [ax1,ax2,ax3,ax4]:
        plt.setp(a.get_xticklabels(), visible=False)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── MESSAGES ──────────────────────────────────────────────────────────────────

SIGNAL_EMOJI = {"STRONG_BUY":"🚀🟢","BUY":"🟢","HOLD":"⏸","SELL":"🔴","STRONG_SELL":"💀🔴"}
SIGNAL_RU    = {"STRONG_BUY":"СИЛЬНАЯ ПОКУПКА","BUY":"ПОКУПКА",
                "HOLD":"УДЕРЖАНИЕ","SELL":"ПРОДАЖА","STRONG_SELL":"СИЛЬНАЯ ПРОДАЖА"}

def fmt_price(p):
    if p >= 1000: return f"${p:,.2f}"
    if p >= 1:    return f"${p:.4f}"
    return f"${p:.6f}"

def build_signal_msg(symbol, res, change_24h):
    s = res["signal"]
    name = symbol.replace("USDT","")
    now  = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    ch_str = f"{'📈' if change_24h>=0 else '📉'} {abs(change_24h):.2f}%"
    reasons = res["reasons_buy"] if "BUY" in s else res["reasons_sell"]
    total = res["buy_score"] + res["sell_score"]

    lines = [
        f"{SIGNAL_EMOJI.get(s,'')} *{name}/USDT — {SIGNAL_RU.get(s,s)}*",
        f"",
        f"💵 Цена: `{fmt_price(res['price'])}`   {ch_str}",
        f"🕐 `{now}`  |  1H + 4H",
        f"",
        f"📊 *Индикаторы (13):*",
        f"  RSI: `{res['rsi']:.1f}`" +
            (" 🔴" if res['rsi']>70 else " 🟢" if res['rsi']<30 else " ⚪"),
        f"  Stoch RSI: K=`{res['stoch_k']:.0f}` D=`{res['stoch_d']:.0f}`",
        f"  Williams %R: `{res['williams_r']:.0f}`",
        f"  CCI: `{res['cci']:.0f}`",
        f"  MACD: `{res['macd_hist']:+.5f}`",
        f"  EMA 9/21/50: `{fmt_price(res['ema9'])}` / `{fmt_price(res['ema21'])}` / `{fmt_price(res['ema50'])}`",
        f"  VWAP: `{fmt_price(res['vwap'])}`",
        f"  Ichimoku: {'☁️ выше облака' if res['above_cloud'] else '☁️ ниже облака' if res['below_cloud'] else '☁️ в облаке'}",
    ]
    if res["support"]:    lines.append(f"  Поддержка: `{fmt_price(res['support'][-1])}`")
    if res["resistance"]: lines.append(f"  Сопротивление: `{fmt_price(res['resistance'][0])}`")
    if res["vol_spike"]:  lines.append(f"  ⚡ Всплеск объёма!")

    # Сентимент
    if res["sentiment_score"] != 0:
        sent_emoji = "🟢" if res["sentiment_score"] > 0 else "🔴"
        lines += ["", f"🧠 *AI Сентимент новостей:* {sent_emoji} `{res['sentiment_score']:+d}/5`"]
        if sentiment_cache.get("summary"):
            lines.append(f"  _{sentiment_cache['summary']}_")

    if reasons:
        lines += ["", "📋 *Почему этот сигнал:*"]
        for r in reasons[:5]:
            lines.append(f"  ▸ {r}")

    if s != "HOLD":
        action = "покупки" if "BUY" in s else "продажи"
        lines += [
            "", f"🎯 *Цели для {action}:*",
            f"  🟢 TP1: `{fmt_price(res['tp1'])}`",
            f"  🟢 TP2: `{fmt_price(res['tp2'])}`",
            f"  🛑 SL:  `{fmt_price(res['sl'])}`",
            f"  📐 ATR: `{fmt_price(res['atr'])}`",
        ]

    lines += [
        "", f"⚡ Сила: {res['buy_score']}🟢 / {res['sell_score']}🔴 из {total}",
        "", "_Не является финансовым советом. DYOR._"
    ]
    return "\n".join(lines)

# ── FEAR & GREED ──────────────────────────────────────────────────────────────

def fear_greed_emoji(val):
    if val is None: return ""
    if val <= 20: return "😱 Extreme Fear"
    if val <= 40: return "😨 Fear"
    if val <= 60: return "😐 Neutral"
    if val <= 80: return "😏 Greed"
    return "🤑 Extreme Greed"

# ── EDUCATIONAL POSTS ─────────────────────────────────────────────────────────

EDUCATIONAL_POSTS = [
    ("📚 Что такое RSI?",
     "RSI — индикатор силы тренда от 0 до 100.\n\n"
     "  ▸ Ниже 30 — перепродан (возможен отскок)\n"
     "  ▸ Выше 70 — перекуплен (возможна коррекция)\n\n"
     "💡 В v4.0 мы используем RSI вместе с 12 другими индикаторами."),
    ("📚 Stochastic RSI — улучшенный RSI",
     "Stochastic RSI = RSI внутри RSI. Более чувствителен к изменениям.\n\n"
     "  ▸ K и D линии — как MACD но для RSI\n"
     "  ▸ K пересекает D снизу вверх ниже 20 = BUY\n"
     "  ▸ K пересекает D сверху вниз выше 80 = SELL\n\n"
     "💡 Реагирует на смену тренда быстрее обычного RSI."),
    ("📚 Williams %R — индикатор разворота",
     "Williams %R показывает положение цены относительно диапазона.\n\n"
     "  ▸ От 0 до -100\n"
     "  ▸ Ниже -80 = перепродан (BUY зона)\n"
     "  ▸ Выше -20 = перекуплен (SELL зона)\n\n"
     "💡 Хорошо работает для определения точек разворота."),
    ("📚 CCI — Commodity Channel Index",
     "CCI показывает насколько цена отклонилась от среднего.\n\n"
     "  ▸ Ниже -100 = перепродан\n"
     "  ▸ Выше +100 = перекуплен\n"
     "  ▸ Около 0 = нейтрально\n\n"
     "💡 Особенно полезен при боковом рынке."),
    ("📚 OBV — давление покупателей",
     "OBV (On-Balance Volume) суммирует объём в направлении тренда.\n\n"
     "  ▸ OBV растёт = покупатели активны = подтверждает рост\n"
     "  ▸ OBV падает = продавцы активны = подтверждает падение\n\n"
     "💡 Расхождение цены и OBV — ранний сигнал разворота."),
    ("📚 VWAP — справедливая цена дня",
     "VWAP = средневзвешенная по объёму цена.\n\n"
     "  ▸ Цена выше VWAP = покупатели контролируют\n"
     "  ▸ Цена ниже VWAP = продавцы контролируют\n\n"
     "💡 Институциональные трейдеры используют VWAP как ориентир."),
    ("📚 Ichimoku Cloud — всё в одном",
     "Ichimoku — японская система из 5 линий.\n\n"
     "  ▸ Цена выше облака = бычий тренд\n"
     "  ▸ Цена ниже облака = медвежий тренд\n"
     "  ▸ Tenkan пересекает Kijun снизу = BUY\n\n"
     "💡 Один индикатор заменяет несколько. Лучше на 4H и 1D."),
    ("📚 AI Сентимент — новое в v4.0",
     "В v4.0 каждый сигнал учитывает новостной фон.\n\n"
     "🧠 Claude AI анализирует последние новости:\n"
     "  ▸ Позитивный сентимент = +1-2 к buy_score\n"
     "  ▸ Негативный сентимент = +1-2 к sell_score\n\n"
     "Примеры влияния:\n"
     "  ▸ 'ФРС снижает ставку' → позитивно для крипты\n"
     "  ▸ 'SEC подаёт иск на биржу' → негативно\n"
     "  ▸ 'BTC ETF одобрен' → очень позитивно"),
    ("📚 Как ФРС влияет на крипту?",
     "ФРС — главный регулятор мировой экономики.\n\n"
     "📉 Повышение ставки → крипта падает\n"
     "📈 Снижение ставки → крипта растёт\n\n"
     "💡 Следи за заседаниями ФРС — они двигают рынок."),
]

edu_index = 0

async def send_educational_post(bot, session):
    global edu_index
    title, text = EDUCATIONAL_POSTS[edu_index % len(EDUCATIONAL_POSTS)]
    edu_index += 1
    fg_val, _ = await fetch_fear_greed(session)
    fg_str = f"\n\n📊 Fear & Greed: `{fg_val}` — {fear_greed_emoji(fg_val)}" if fg_val else ""
    await bot.send_message(CHANNEL_ID, f"{title}\n\n{text}{fg_str}", parse_mode=ParseMode.MARKDOWN)

# ── PRICE LEVEL ALERTS ────────────────────────────────────────────────────────

async def check_price_levels(bot, session):
    for symbol, levels in PRICE_LEVELS.items():
        try:
            ticker = await fetch_ticker(session, symbol)
            price  = float(ticker["lastPrice"])
            name   = symbol.replace("USDT","")
            for level in levels:
                key  = f"{symbol}_{level}"
                prev = last_price_alerts.get(key, 0)
                if prev != 0:
                    crossed = (prev < level <= price) or (prev > level >= price)
                    if crossed:
                        direction = "пробил вверх 🚀" if price > level else "пробил вниз 📉"
                        msg = (f"🔔 *Ценовой алерт!*\n\n"
                               f"*{name}* {direction} уровень `${level:,}`\n"
                               f"Цена: `${price:,.2f}`\n\n"
                               f"_Ключевые уровни — поддержка/сопротивление_")
                        await bot.send_message(CHANNEL_ID, msg, parse_mode=ParseMode.MARKDOWN)
                        await asyncio.sleep(1)
                last_price_alerts[key] = price
        except Exception as e:
            logger.error(f"Price level {symbol}: {e}")

# ── BTC DAILY ANALYSIS ────────────────────────────────────────────────────────

async def send_btc_analysis(bot, session):
    df1h  = await fetch_klines(session, "BTCUSDT", "1h", 150)
    df4h  = await fetch_klines(session, "BTCUSDT", "4h", 100)
    df1d  = await fetch_klines(session, "BTCUSDT", "1d", 30)
    ticker = await fetch_ticker(session, "BTCUSDT")
    fg_val, _ = await fetch_fear_greed(session)
    btc_dom   = await fetch_btc_dominance(session)
    if df1h is None: return

    sent = sentiment_cache.get("score", 0)
    res  = analyze(df1h, df4h, sent)
    price = res["price"]
    rsi   = res["rsi"]
    ch24  = float(ticker.get("priceChangePercent", 0))
    week_change = 0
    if df1d is not None and len(df1d) >= 7:
        week_change = ((df1d["close"].iloc[-1] / df1d["close"].iloc[-7]) - 1) * 100

    if res["ema9"] > res["ema21"] > res["ema50"]:
        trend = "🐂 Бычий тренд"
        trend_desc = "EMA выстроены вверх — покупатели контролируют"
    elif res["ema9"] < res["ema21"] < res["ema50"]:
        trend = "🐻 Медвежий тренд"
        trend_desc = "EMA выстроены вниз — продавцы давят"
    else:
        trend = "↔️ Боковик"
        trend_desc = "EMA смешаны — рынок в нерешительности"

    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    lines = [
        "₿ *ЕЖЕДНЕВНЫЙ РАЗБОР BTC*",
        f"_{now}_", "",
        f"💵 `${price:,.2f}`   📈 24h: `{ch24:+.2f}%`   📅 7d: `{week_change:+.2f}%`",
        "", f"📊 *Тренд: {trend}*", f"_{trend_desc}_", "",
        f"🔢 *13 Индикаторов:*",
        f"  RSI: `{rsi:.1f}` | Stoch K/D: `{res['stoch_k']:.0f}`/`{res['stoch_d']:.0f}`",
        f"  Williams %R: `{res['williams_r']:.0f}` | CCI: `{res['cci']:.0f}`",
        f"  MACD: `{res['macd_hist']:+.2f}`",
        f"  EMA 9/21/50: `${res['ema9']:,.0f}`/`${res['ema21']:,.0f}`/`${res['ema50']:,.0f}`",
        f"  VWAP: `${res['vwap']:,.0f}`",
        f"  Ichimoku: {'выше облака ☁️' if res['above_cloud'] else 'ниже облака ☁️' if res['below_cloud'] else 'в облаке ☁️'}",
    ]
    sup_str = f"`${res['support'][-1]:,.0f}`" if res["support"] else "н/д"
    res_str = f"`${res['resistance'][0]:,.0f}`" if res["resistance"] else "н/д"
    lines += ["", f"📐 Поддержка: {sup_str}  |  Сопротивление: {res_str}", ""]

    if fg_val:  lines.append(f"😱 Fear & Greed: `{fg_val}` — {fear_greed_emoji(fg_val)}")
    if btc_dom: lines.append(f"₿ BTC Dominance: `{btc_dom}%`")
    if sent != 0:
        lines.append(f"🧠 AI Сентимент: `{sent:+d}/5`")

    lines += [
        "", f"🎯 Сигнал: *{res['signal']}*",
        f"  TP1: `${res['tp1']:,.0f}`  TP2: `${res['tp2']:,.0f}`  SL: `${res['sl']:,.0f}`",
        "", "_Обновляется ежедневно. DYOR._",
    ]
    chart = build_chart(res, "BTCUSDT")
    await bot.send_photo(CHANNEL_ID, photo=chart,
                         caption="\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# ── SCREENER ──────────────────────────────────────────────────────────────────

async def send_screener(bot, session):
    sent = sentiment_cache.get("score", 0)
    scores = []
    for symbol in WATCHLIST:
        try:
            df1h = await fetch_klines(session, symbol, "1h", 150)
            df4h = await fetch_klines(session, symbol, "4h", 60)
            if df1h is None or len(df1h) < 60: continue
            res = analyze(df1h, df4h, sent)
            potential = 0
            direction = "neutral"
            if res["buy_score"] >= 5:
                potential = res["buy_score"]; direction = "buy"
            elif res["sell_score"] >= 5:
                potential = res["sell_score"]; direction = "sell"
            if potential > 0:
                ticker = await fetch_ticker(session, symbol)
                scores.append({
                    "symbol": symbol, "res": res,
                    "potential": potential, "direction": direction,
                    "ch24": float(ticker.get("priceChangePercent", 0)),
                })
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"Screener {symbol}: {e}")

    if not scores: return
    scores.sort(key=lambda x: x["potential"], reverse=True)
    top3 = scores[:3]

    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    lines = ["🔍 *СКРИНЕР — ТОП ПАРЫ*", f"_{now}_", "",
             "_Пары с наибольшим потенциалом:_", ""]

    for i, item in enumerate(top3, 1):
        name = item["symbol"].replace("USDT","")
        res  = item["res"]
        p    = res["price"]
        p_str = fmt_price(p)
        dir_emoji = "🟢 BUY" if item["direction"] == "buy" else "🔴 SELL"
        ch_str = f"{'📈' if item['ch24']>=0 else '📉'} {abs(item['ch24']):.1f}%"
        lines += [
            f"*{i}. {name}/USDT* — {dir_emoji}",
            f"   Цена: `{p_str}`  {ch_str}",
            f"   Сила сигнала: `{item['potential']}/20`",
            f"   RSI: `{res['rsi']:.0f}` | Stoch: `{res['stoch_k']:.0f}`",
            f"   TP1: `{fmt_price(res['tp1'])}`  SL: `{fmt_price(res['sl'])}`",
            "",
        ]
    lines.append("_Скринер каждые 6 часов. Не финансовый совет._")
    await bot.send_message(CHANNEL_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# ── DAILY DIGEST ──────────────────────────────────────────────────────────────

async def send_daily_digest(bot, session):
    tickers = await fetch_all_tickers(session)
    pairs   = {t["symbol"]: t for t in tickers if t["symbol"] in WATCHLIST}
    fg_val, _ = await fetch_fear_greed(session)
    btc_dom   = await fetch_btc_dominance(session)
    rows = []
    for sym in WATCHLIST:
        t = pairs.get(sym)
        if not t: continue
        rows.append((sym.replace("USDT",""), float(t["lastPrice"]),
                     float(t["priceChangePercent"]), float(t["quoteVolume"])))
    rows.sort(key=lambda r: abs(r[2]), reverse=True)

    lines = ["🌅 *УТРЕННИЙ ДАЙДЖЕСТ*",
             f"_{datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}_", ""]
    if fg_val:  lines.append(f"😱 Fear & Greed: *{fg_val}* — {fear_greed_emoji(fg_val)}")
    if btc_dom: lines.append(f"₿ BTC Dominance: *{btc_dom}%*")
    sent = sentiment_cache.get("score", 0)
    if sent != 0:
        lines.append(f"🧠 AI Сентимент: `{sent:+d}/5` {'🟢' if sent>0 else '🔴'}")
    lines += ["", "```", f"{'Пара':<8} {'Цена':>12} {'24h':>8} {'Объём':>12}", "─"*44]
    for name, price, ch, vol in rows:
        arrow = "▲" if ch >= 0 else "▼"
        p_str = f"${price:>10,.2f}" if price >= 1 else f"${price:>10.4f}"
        lines.append(f"{name:<8} {p_str} {arrow}{abs(ch):>5.1f}% ${vol/1e6:>7.1f}M")
    lines.append("```")
    gainer = max(rows, key=lambda r: r[2])
    loser  = min(rows, key=lambda r: r[2])
    lines += ["", f"🏆 Лидер роста:   *{gainer[0]}* `+{gainer[2]:.2f}%`",
              f"📉 Лидер падения: *{loser[0]}*  `{loser[2]:.2f}%`",
              "", "_Следующий дайджест через 24 часа_"]
    await bot.send_message(CHANNEL_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# ── SIGNAL CHECK ──────────────────────────────────────────────────────────────

def should_alert(symbol, new_signal):
    if new_signal == "HOLD": return False
    prev = last_alerts.get(symbol, {})
    if prev.get("signal") == new_signal:
        if (datetime.now(timezone.utc).timestamp() - prev.get("ts", 0)) < 4 * 3600:
            return False
    return True

async def check_signals(bot, session):
    logger.info("Checking signals for %d pairs…", len(WATCHLIST))
    await check_price_levels(bot, session)
    sent = sentiment_cache.get("score", 0)
    for symbol in WATCHLIST:
        try:
            df1h = await fetch_klines(session, symbol, "1h", 150)
            df4h = await fetch_klines(session, symbol, "4h", 60)
            if df1h is None or len(df1h) < 60: continue
            res = analyze(df1h, df4h, sent)
            if not should_alert(symbol, res["signal"]): continue
            ticker = await fetch_ticker(session, symbol)
            change = float(ticker.get("priceChangePercent", 0))
            chart   = build_chart(res, symbol)
            caption = build_signal_msg(symbol, res, change)
            await bot.send_photo(CHANNEL_ID, photo=chart,
                                 caption=caption, parse_mode=ParseMode.MARKDOWN)
            last_alerts[symbol] = {"signal": res["signal"],
                                   "ts": datetime.now(timezone.utc).timestamp()}
            logger.info("Alert: %s → %s (sent=%+d)", symbol, res["signal"], sent)
            await asyncio.sleep(1.5)
        except TelegramError as e:
            logger.error("Telegram %s: %s", symbol, e)
        except Exception as e:
            logger.exception("Error %s: %s", symbol, e)

# ── MAIN LOOPS ────────────────────────────────────────────────────────────────

async def signal_loop(bot, session):
    while True:
        await check_signals(bot, session)
        await asyncio.sleep(CHECK_INTERVAL)

async def news_loop(bot, session):
    await asyncio.sleep(600)
    while True:
        try:
            articles = await fetch_world_news(session)
            titles = [a.get("title","") for a in articles]
            await analyze_sentiment_ai(session, titles)
            await send_news_digest(bot, session)
        except Exception as e: logger.error("News: %s", e)
        await asyncio.sleep(NEWS_INTERVAL)

async def digest_loop(bot, session):
    while True:
        try: await send_daily_digest(bot, session)
        except Exception as e: logger.error("Digest: %s", e)
        await asyncio.sleep(DIGEST_INTERVAL)

async def btc_analysis_loop(bot, session):
    await asyncio.sleep(300)
    while True:
        try: await send_btc_analysis(bot, session)
        except Exception as e: logger.error("BTC: %s", e)
        await asyncio.sleep(BTC_ANALYSIS_INTERVAL)

async def screener_loop(bot, session):
    await asyncio.sleep(900)
    while True:
        try: await send_screener(bot, session)
        except Exception as e: logger.error("Screener: %s", e)
        await asyncio.sleep(SCREENER_INTERVAL)

async def education_loop(bot, session):
    await asyncio.sleep(1800)
    while True:
        try: await send_educational_post(bot, session)
        except Exception as e: logger.error("Education: %s", e)
        await asyncio.sleep(EDUCATION_INTERVAL)

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    me  = await bot.get_me()
    logger.info("Bot v4.0 started: @%s", me.username)
    async with aiohttp.ClientSession() as session:
        await bot.send_message(
            CHANNEL_ID,
            "🤖 *Crypto Signals Bot v4.0 запущен!*\n\n"
            "📊 13 индикаторов:\n"
            "RSI, Stoch RSI, Williams %R, CCI,\n"
            "MACD, EMA 9/21/50, Bollinger Bands,\n"
            "OBV, VWAP, Ichimoku, ATR, Объём, Поддержка/Сопротивление\n\n"
            "🧠 AI Сентимент новостей влияет на сигналы\n"
            "📰 Мировые новости каждые 2 часа\n"
            "₿ Ежедневный разбор BTC\n"
            "🔍 Скринер топ-3 пары каждые 6 часов",
            parse_mode=ParseMode.MARKDOWN,
        )
        await asyncio.gather(
            signal_loop(bot, session),
            news_loop(bot, session),
            digest_loop(bot, session),
            btc_analysis_loop(bot, session),
            screener_loop(bot, session),
            education_loop(bot, session),
        )

if __name__ == "__main__":
    asyncio.run(main())
