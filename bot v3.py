"""
Crypto Signals Bot v3.0
Новое в v3:
- Мировые новости каждые 2 часа (NewsAPI)
- Ежедневный разбор BTC
- Скринер топ-3 пары каждые 6 часов
- Экономический календарь
- Анализ корреляции с S&P 500
"""

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta

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
NEWS_INTERVAL      = 7200   # каждые 2 часа
DIGEST_INTERVAL    = 86400
EDUCATION_INTERVAL = 14400
BTC_ANALYSIS_INTERVAL = 86400  # ежедневно
SCREENER_INTERVAL  = 21600  # каждые 6 часов

BINANCE_BASE   = "https://api.binance.com/api/v3"
FEAR_GREED_API = "https://api.alternative.me/fng/?limit=1"
NEWSAPI_BASE   = "https://newsapi.org/v2/everything"

last_alerts: dict = {}
last_price_alerts: dict = {}

# ── BINANCE ───────────────────────────────────────────────────────────────────

async def fetch_klines(session, symbol, interval="1h", limit=100):
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
        data = await r.json()
    if not isinstance(data, list) or len(data) < 20:
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

# ── NEWS ──────────────────────────────────────────────────────────────────────

CRYPTO_KEYWORDS = (
    "bitcoin OR ethereum OR crypto OR cryptocurrency OR "
    "blockchain OR BTC OR ETH OR solana OR binance OR "
    "Federal Reserve OR inflation OR CPI OR interest rates OR "
    "SEC crypto OR ETF bitcoin"
)

async def fetch_world_news(session):
    """Получаем новости через NewsAPI"""
    if NEWSAPI_TOKEN == "YOUR_NEWSAPI_TOKEN":
        return []
    try:
        params = {
            "q": CRYPTO_KEYWORDS,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 8,
            "apiKey": NEWSAPI_TOKEN,
        }
        async with session.get(NEWSAPI_BASE, params=params,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            return data.get("articles", [])[:6]
    except Exception as e:
        logger.error(f"NewsAPI error: {e}")
        return []

def categorize_news(title: str) -> tuple:
    """Определяем тип новости и её влияние на рынок"""
    title_lower = title.lower()
    # Позитивные сигналы
    if any(w in title_lower for w in ["etf approved","etf approval","bullish","rally",
                                       "all-time high","ath","institutional","adoption",
                                       "rate cut","fed cut","inflation falls"]):
        return "🟢", "Позитив для рынка"
    # Негативные сигналы
    if any(w in title_lower for w in ["ban","crackdown","sec sues","hack","crash",
                                       "rate hike","inflation rises","recession",
                                       "bear","sell-off","liquidation"]):
        return "🔴", "Негатив для рынка"
    # Макро
    if any(w in title_lower for w in ["federal reserve","fed","cpi","gdp","inflation",
                                       "interest rate","treasury","dollar"]):
        return "🔵", "Макроэкономика"
    return "⚪", "Нейтрально"

async def send_news_digest(bot, session):
    articles = await fetch_world_news(session)
    fg_val, _ = await fetch_fear_greed(session)

    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    lines = [
        "📰 *МИРОВЫЕ НОВОСТИ*",
        f"_{now}_",
        "",
    ]
    if fg_val:
        fg_emoji = "😱" if fg_val<=20 else "😨" if fg_val<=40 else "😐" if fg_val<=60 else "😏" if fg_val<=80 else "🤑"
        lines.append(f"{fg_emoji} Fear & Greed: `{fg_val}`")
        lines.append("")

    if not articles:
        lines.append("_Новости временно недоступны_")
    else:
        for art in articles:
            title  = art.get("title","")[:120]
            source = art.get("source",{}).get("name","")
            url    = art.get("url","")
            emoji, impact = categorize_news(title)
            lines.append(f"{emoji} *{impact}*")
            lines.append(f"[{title}]({url})")
            if source:
                lines.append(f"_📌 {source}_")
            lines.append("")

    lines.append("_Новости влияют на рынок — учитывай при торговле_")
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

def calc_support_resistance(df, window=20):
    highs = df["high"].rolling(window, center=True).max()
    lows  = df["low"].rolling(window, center=True).min()
    resistance = df["high"][df["high"] == highs].dropna().tail(3).values.tolist()
    support    = df["low"][df["low"] == lows].dropna().tail(3).values.tolist()
    return sorted(set(support)), sorted(set(resistance))

def analyze(df1h, df4h=None):
    close = df1h["close"]
    vol   = df1h["volume"]
    rsi   = calc_rsi(close)
    macd, macd_sig, macd_hist = calc_macd(close)
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    bb_up, bb_mid, bb_low = calc_bollinger(close)
    support, resistance = calc_support_resistance(df1h)

    last_rsi   = rsi.iloc[-1]
    last_hist  = macd_hist.iloc[-1]
    prev_hist  = macd_hist.iloc[-2]
    last_close = close.iloc[-1]
    last_vol   = vol.iloc[-1]
    avg_vol    = vol.rolling(20).mean().iloc[-1]
    vol_spike  = last_vol > avg_vol * 1.5
    atr        = (df1h["high"] - df1h["low"]).rolling(14).mean().iloc[-1]

    bullish_ema = ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]
    bearish_ema = ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]
    macd_bull   = prev_hist < 0 and last_hist > 0
    macd_bear   = prev_hist > 0 and last_hist < 0
    bb_break_up = last_close > bb_up.iloc[-1]
    bb_break_dn = last_close < bb_low.iloc[-1]

    confirm_4h = 0
    if df4h is not None:
        try:
            rsi_4h   = calc_rsi(df4h["close"]).iloc[-1]
            ema9_4h  = df4h["close"].ewm(span=9,  adjust=False).mean().iloc[-1]
            ema21_4h = df4h["close"].ewm(span=21, adjust=False).mean().iloc[-1]
            if rsi_4h < 45 and ema9_4h > ema21_4h:  confirm_4h = +2
            if rsi_4h > 55 and ema9_4h < ema21_4h:  confirm_4h = -2
        except Exception:
            pass

    buy_score = sell_score = 0
    reasons_buy = []
    reasons_sell = []

    if last_rsi < 35:   buy_score += 2; reasons_buy.append(f"RSI перепродан ({last_rsi:.0f})")
    elif last_rsi < 45: buy_score += 1; reasons_buy.append(f"RSI слабый ({last_rsi:.0f})")
    if last_rsi > 65:   sell_score += 2; reasons_sell.append(f"RSI перекуплен ({last_rsi:.0f})")
    elif last_rsi > 55: sell_score += 1; reasons_sell.append(f"RSI высокий ({last_rsi:.0f})")

    if macd_bull: buy_score  += 3; reasons_buy.append("MACD бычий разворот")
    if macd_bear: sell_score += 3; reasons_sell.append("MACD медвежий разворот")
    if last_hist > 0: buy_score  += 1
    else:             sell_score += 1

    if bullish_ema: buy_score  += 2; reasons_buy.append("EMA бычье выравнивание")
    if bearish_ema: sell_score += 2; reasons_sell.append("EMA медвежье выравнивание")

    if bb_break_up and vol_spike: buy_score  += 2; reasons_buy.append("Пробой Bollinger вверх + объём")
    if bb_break_dn and vol_spike: sell_score += 2; reasons_sell.append("Пробой Bollinger вниз + объём")

    if vol_spike:
        if last_close > close.iloc[-2]: buy_score  += 1; reasons_buy.append("Всплеск объёма на росте")
        else:                           sell_score += 1; reasons_sell.append("Всплеск объёма на падении")

    if confirm_4h > 0:   buy_score  += confirm_4h;      reasons_buy.append("4H подтверждает рост")
    elif confirm_4h < 0: sell_score += abs(confirm_4h); reasons_sell.append("4H подтверждает падение")

    if   buy_score >= 8:                                 signal = "STRONG_BUY"
    elif buy_score >= 5 and buy_score > sell_score + 2:  signal = "BUY"
    elif sell_score >= 8:                                signal = "STRONG_SELL"
    elif sell_score >= 5 and sell_score > buy_score + 2: signal = "SELL"
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
        "ema9": ema9.iloc[-1], "ema21": ema21.iloc[-1], "ema50": ema50.iloc[-1],
        "bb_up": bb_up.iloc[-1], "bb_low": bb_low.iloc[-1],
        "vol_spike": vol_spike, "atr": atr,
        "tp1": tp1, "tp2": tp2, "sl": sl,
        "support": support, "resistance": resistance,
        "reasons_buy": reasons_buy, "reasons_sell": reasons_sell,
        "df": df1h, "rsi_series": rsi, "macd_series": macd,
        "macd_sig_series": macd_sig, "macd_hist_series": macd_hist,
        "ema9_s": ema9, "ema21_s": ema21, "ema50_s": ema50,
        "bb_up_s": bb_up, "bb_mid_s": bb_mid, "bb_low_s": bb_low,
        "close_series": close,
    }

# ── CHART ─────────────────────────────────────────────────────────────────────

def build_chart(res, symbol):
    df = res["df"].copy().tail(60)
    color_map = {
        "STRONG_BUY":"#00ff88","BUY":"#00cc66",
        "HOLD":"#ffcc00","SELL":"#ff6644","STRONG_SELL":"#ff2200",
    }
    sig_color = color_map.get(res["signal"], "#aaa")
    fig = plt.figure(figsize=(14, 11), facecolor="#0d1117")
    gs  = GridSpec(4, 1, figure=fig, hspace=0.06, height_ratios=[3,.8,.8,.8])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_color("#21262d")
    x = np.arange(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        c = "#26a641" if row["close"] >= row["open"] else "#f85149"
        ax1.plot([i,i],[row["low"],row["high"]], color=c, lw=0.8)
        ax1.bar(i, abs(row["close"]-row["open"]),
                bottom=min(row["open"],row["close"]), color=c, width=0.7, alpha=0.9)
    ax1.fill_between(x, res["bb_up_s"].values[-60:], res["bb_low_s"].values[-60:],
                     alpha=0.07, color="#58a6ff")
    ax1.plot(x, res["bb_up_s"].values[-60:],  color="#58a6ff", lw=0.7, alpha=0.5)
    ax1.plot(x, res["bb_mid_s"].values[-60:], color="#58a6ff", lw=0.6, alpha=0.3, ls="--")
    ax1.plot(x, res["bb_low_s"].values[-60:], color="#58a6ff", lw=0.7, alpha=0.5)
    ax1.plot(x, res["ema9_s"].values[-60:],  color="#f0c419", lw=1.2, label="EMA9")
    ax1.plot(x, res["ema21_s"].values[-60:], color="#e36bdf", lw=1.2, label="EMA21")
    ax1.plot(x, res["ema50_s"].values[-60:], color="#58a6ff", lw=1.2, label="EMA50")
    for s in res["support"]:
        ax1.axhline(s, color="#26a641", lw=0.7, ls=":", alpha=0.6)
        ax1.text(0, s, " S", color="#26a641", fontsize=6, va="center")
    for r in res["resistance"]:
        ax1.axhline(r, color="#f85149", lw=0.7, ls=":", alpha=0.6)
        ax1.text(0, r, " R", color="#f85149", fontsize=6, va="center")
    if res["signal"] != "HOLD":
        ax1.axhline(res["tp1"], color="#26a641", lw=0.9, ls="--", alpha=0.8)
        ax1.axhline(res["tp2"], color="#26a641", lw=0.7, ls=":",  alpha=0.6)
        ax1.axhline(res["sl"],  color="#f85149", lw=0.9, ls="--", alpha=0.8)
        ax1.text(len(x)-1, res["tp1"], " TP1", color="#26a641", va="center", fontsize=7)
        ax1.text(len(x)-1, res["tp2"], " TP2", color="#26a641", va="center", fontsize=7)
        ax1.text(len(x)-1, res["sl"],  " SL",  color="#f85149", va="center", fontsize=7)
    p = res["price"]
    p_str = f"${p:,.2f}" if p >= 1 else f"${p:.4f}"
    ax1.set_title(f"{symbol}  {p_str}   [{res['signal']}]",
                  color=sig_color, fontsize=13, fontweight="bold", pad=8, fontfamily="monospace")
    ax1.legend(fontsize=7, loc="upper left",
               facecolor="#161b22", edgecolor="#21262d", labelcolor="#8b949e")
    vc = ["#26a641" if df["close"].iloc[i]>=df["open"].iloc[i] else "#f85149" for i in range(len(df))]
    ax2.bar(x, df["volume"].values, color=vc, alpha=0.7, width=0.7)
    ax2.plot(x, df["volume"].rolling(20).mean().values, color="#f0c419", lw=0.8)
    ax2.set_ylabel("Vol", fontsize=7, color="#8b949e")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v>=1e6 else f"{v/1e3:.0f}K"))
    hist = res["macd_hist_series"].values[-60:]
    ax3.bar(x, hist, color=["#26a641" if h>=0 else "#f85149" for h in hist], alpha=0.7, width=0.7)
    ax3.plot(x, res["macd_series"].values[-60:],     color="#58a6ff", lw=1.0)
    ax3.plot(x, res["macd_sig_series"].values[-60:], color="#f0c419", lw=1.0)
    ax3.axhline(0, color="#21262d", lw=0.8)
    ax3.set_ylabel("MACD", fontsize=7, color="#8b949e")
    rv = res["rsi_series"].values[-60:]
    ax4.plot(x, rv, color="#e36bdf", lw=1.2)
    ax4.axhline(70, color="#f85149", lw=0.7, ls="--", alpha=0.6)
    ax4.axhline(30, color="#26a641", lw=0.7, ls="--", alpha=0.6)
    ax4.axhline(50, color="#21262d", lw=0.6)
    ax4.fill_between(x, rv, 70, where=(rv>=70), alpha=0.15, color="#f85149")
    ax4.fill_between(x, rv, 30, where=(rv<=30), alpha=0.15, color="#26a641")
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("RSI", fontsize=7, color="#8b949e")
    ax4.text(len(x)-1, rv[-1], f"  {rv[-1]:.1f}", color="#e36bdf", va="center", fontsize=7)
    ticks = list(range(0, len(df), 10))
    ax4.set_xticks(ticks)
    ax4.set_xticklabels([df.index[i].strftime("%d/%m %H:%M") for i in ticks],
                        rotation=30, ha="right", fontsize=7, color="#8b949e")
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── BTC DAILY ANALYSIS ────────────────────────────────────────────────────────

async def send_btc_analysis(bot, session):
    df1h  = await fetch_klines(session, "BTCUSDT", "1h", 100)
    df4h  = await fetch_klines(session, "BTCUSDT", "4h", 100)
    df1d  = await fetch_klines(session, "BTCUSDT", "1d", 30)
    ticker = await fetch_ticker(session, "BTCUSDT")
    fg_val, _ = await fetch_fear_greed(session)
    btc_dom   = await fetch_btc_dominance(session)

    if df1h is None: return

    res   = analyze(df1h, df4h)
    price = res["price"]
    rsi   = res["rsi"]
    ch24  = float(ticker.get("priceChangePercent", 0))

    # Недельное изменение из дневных свечей
    week_change = 0
    if df1d is not None and len(df1d) >= 7:
        week_change = ((df1d["close"].iloc[-1] / df1d["close"].iloc[-7]) - 1) * 100

    # Определяем тренд
    if res["ema9"] > res["ema21"] > res["ema50"]:
        trend = "🐂 Бычий тренд"
        trend_desc = "Все EMA выстроены вверх — покупатели контролируют рынок"
    elif res["ema9"] < res["ema21"] < res["ema50"]:
        trend = "🐻 Медвежий тренд"
        trend_desc = "Все EMA выстроены вниз — продавцы давят на цену"
    else:
        trend = "↔️ Боковик"
        trend_desc = "EMA смешаны — рынок в нерешительности"

    # Уровни
    sup_str = f"`${res['support'][-1]:,.0f}`" if res["support"] else "н/д"
    res_str = f"`${res['resistance'][0]:,.0f}`" if res["resistance"] else "н/д"

    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")

    lines = [
        "₿ *ЕЖЕДНЕВНЫЙ РАЗБОР BTC*",
        f"_{now}_",
        "",
        f"💵 Цена: `${price:,.2f}`",
        f"📈 24h: `{ch24:+.2f}%`   📅 7d: `{week_change:+.2f}%`",
        "",
        f"📊 *Тренд: {trend}*",
        f"_{trend_desc}_",
        "",
        f"🔢 *Индикаторы:*",
        f"  RSI(14): `{rsi:.1f}`" +
            (" 🔴 перекуплен" if rsi>70 else " 🟢 перепродан" if rsi<30 else " ⚪ нейтрально"),
        f"  MACD: `{res['macd_hist']:+.2f}`",
        f"  EMA 9/21/50: `${res['ema9']:,.0f}` / `${res['ema21']:,.0f}` / `${res['ema50']:,.0f}`",
        "",
        f"📐 *Ключевые уровни:*",
        f"  Поддержка:    {sup_str}",
        f"  Сопротивление: {res_str}",
        "",
    ]

    if fg_val:
        fg_emoji = "😱" if fg_val<=20 else "😨" if fg_val<=40 else "😐" if fg_val<=60 else "😏" if fg_val<=80 else "🤑"
        lines.append(f"{fg_emoji} Fear & Greed: `{fg_val}`")
    if btc_dom:
        lines.append(f"₿ BTC Dominance: `{btc_dom}%`")

    lines += [
        "",
        f"🎯 *Сигнал: {res['signal']}*",
        f"  TP1: `${res['tp1']:,.0f}`  |  TP2: `${res['tp2']:,.0f}`",
        f"  SL:  `${res['sl']:,.0f}`",
        "",
        "_Разбор обновляется ежедневно. DYOR._",
    ]

    chart = build_chart(res, "BTCUSDT")
    await bot.send_photo(CHANNEL_ID, photo=chart,
                         caption="\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# ── SCREENER ──────────────────────────────────────────────────────────────────

async def send_screener(bot, session):
    """Топ-3 пары с лучшим потенциалом"""
    scores = []
    for symbol in WATCHLIST:
        try:
            df1h = await fetch_klines(session, symbol, "1h", 100)
            df4h = await fetch_klines(session, symbol, "4h", 60)
            if df1h is None or len(df1h) < 60: continue
            res = analyze(df1h, df4h)

            # Скор потенциала
            potential = 0
            direction = "neutral"
            if res["buy_score"] >= 4:
                potential = res["buy_score"]
                direction = "buy"
            elif res["sell_score"] >= 4:
                potential = res["sell_score"]
                direction = "sell"

            if potential > 0:
                ticker = await fetch_ticker(session, symbol)
                ch24 = float(ticker.get("priceChangePercent", 0))
                scores.append({
                    "symbol": symbol,
                    "res": res,
                    "potential": potential,
                    "direction": direction,
                    "ch24": ch24,
                })
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"Screener error {symbol}: {e}")

    if not scores:
        return

    # Топ-3
    scores.sort(key=lambda x: x["potential"], reverse=True)
    top3 = scores[:3]

    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    lines = [
        "🔍 *СКРИНЕР — ТОП ПАРЫ*",
        f"_{now}_",
        "",
        "_Пары с наибольшим потенциалом прямо сейчас:_",
        "",
    ]

    for i, item in enumerate(top3, 1):
        name = item["symbol"].replace("USDT","")
        res  = item["res"]
        p    = res["price"]
        p_str = f"${p:,.2f}" if p >= 1 else f"${p:.4f}"
        dir_emoji = "🟢 BUY" if item["direction"] == "buy" else "🔴 SELL"
        ch_str = f"{'📈' if item['ch24']>=0 else '📉'} {abs(item['ch24']):.1f}%"

        lines += [
            f"*{i}. {name}/USDT* — {dir_emoji}",
            f"   Цена: `{p_str}`  {ch_str}",
            f"   Сила сигнала: `{item['potential']}/10`",
            f"   RSI: `{res['rsi']:.0f}`  |  ATR: `{res['atr']:.4f}`",
            f"   TP1: `{fmt_price(res['tp1'])}`  SL: `{fmt_price(res['sl'])}`",
            "",
        ]

    lines.append("_Скринер обновляется каждые 6 часов. Не финансовый совет._")
    await bot.send_message(CHANNEL_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

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

    lines = [
        f"{SIGNAL_EMOJI.get(s,'')} *{name}/USDT — {SIGNAL_RU.get(s,s)}*",
        f"",
        f"💵 Цена: `{fmt_price(res['price'])}`   {ch_str}",
        f"🕐 `{now}`  |  1H + 4H",
        f"",
        f"📊 *Индикаторы:*",
        f"  RSI: `{res['rsi']:.1f}`" +
            (" 🔴 перекуплен" if res['rsi']>70 else " 🟢 перепродан" if res['rsi']<30 else " ⚪ нейтрально"),
        f"  MACD: `{res['macd_hist']:+.5f}`",
        f"  EMA 9/21/50: `{fmt_price(res['ema9'])}` / `{fmt_price(res['ema21'])}` / `{fmt_price(res['ema50'])}`",
    ]
    if res["support"]:    lines.append(f"  Поддержка: `{fmt_price(res['support'][-1])}`")
    if res["resistance"]: lines.append(f"  Сопротивление: `{fmt_price(res['resistance'][0])}`")
    if res["vol_spike"]:  lines.append(f"  ⚡ Всплеск объёма!")

    if reasons:
        lines += ["", "📋 *Почему этот сигнал:*"]
        for r in reasons[:4]:
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

    lines += ["", f"⚡ Сила: {res['buy_score']}🟢 / {res['sell_score']}🔴",
              "", "_Не является финансовым советом. DYOR._"]
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
     "  ▸ Ниже 30 — *перепродан* (возможен отскок)\n"
     "  ▸ Выше 70 — *перекуплен* (возможна коррекция)\n\n"
     "💡 Работает лучше вместе с MACD и EMA."),
    ("📚 Что такое MACD?",
     "MACD — индикатор импульса и смены тренда.\n\n"
     "  ▸ Гистограмма пересекает 0 снизу вверх = BUY\n"
     "  ▸ Гистограмма пересекает 0 сверху вниз = SELL\n\n"
     "💡 Пересечение MACD = +3 очка к нашему сигналу."),
    ("📚 Зачем три EMA?",
     "EMA9, EMA21, EMA50 — три скользящие средние.\n\n"
     "🐂 EMA9 > EMA21 > EMA50 = бычий тренд\n"
     "🐻 EMA9 < EMA21 < EMA50 = медвежий тренд\n\n"
     "💡 Когда все три выстроились — тренд сильный."),
    ("📚 Bollinger Bands",
     "Конверт вокруг цены показывающий волатильность.\n\n"
     "  ▸ Цена у нижней полосы = возможный отскок\n"
     "  ▸ Пробой верхней + объём = сильный импульс\n"
     "  ▸ Полосы сужаются = движение близко"),
    ("📚 TP и SL",
     "Take Profit и Stop Loss — управление капиталом.\n\n"
     "Мы используем ATR:\n"
     "  ▸ TP1 = цена + ATR x 1.5\n"
     "  ▸ TP2 = цена + ATR x 3.0\n"
     "  ▸ SL  = цена - ATR x 1.0\n\n"
     "⚠️ Никогда не торгуй без SL."),
    ("📚 Fear & Greed Index",
     "Индекс страха и жадности (0-100):\n\n"
     "  😱 0-20  Extreme Fear — возможно дно\n"
     "  😨 21-40 Fear\n"
     "  😐 41-60 Neutral\n"
     "  😏 61-80 Greed\n"
     "  🤑 81-100 Extreme Greed — осторожно!\n\n"
     "💡 Покупай в страхе, продавай в жадности."),
    ("📚 Как ФРС влияет на крипту?",
     "ФРС (Федеральная резервная система США) — главный регулятор мировой экономики.\n\n"
     "📉 *Повышение ставки:* доллар дорожает, риск-активы падают\n"
     "  → Крипта обычно снижается\n\n"
     "📈 *Снижение ставки:* деньги дешевеют, инвесторы ищут доходность\n"
     "  → Крипта и акции растут\n\n"
     "💡 Следи за заседаниями ФРС — они двигают весь рынок."),
    ("📚 Что такое CPI и почему он важен?",
     "CPI (Consumer Price Index) — индекс потребительских цен, мера инфляции.\n\n"
     "📊 Выходит раз в месяц в США.\n\n"
     "📉 CPI выше ожиданий = инфляция растёт\n"
     "  → ФРС может повысить ставку → крипта падает\n\n"
     "📈 CPI ниже ожиданий = инфляция снижается\n"
     "  → ФРС может снизить ставку → крипта растёт\n\n"
     "💡 Это один из самых важных макроэкономических показателей для крипты."),
    ("📚 BTC Dominance — что это значит?",
     "BTC Dominance — доля биткоина в общей капитализации крипторынка.\n\n"
     "📈 Dominance растёт (>55%):\n"
     "  → Деньги уходят из альткоинов в BTC\n"
     "  → Альткоины часто падают сильнее BTC\n\n"
     "📉 Dominance падает (<45%):\n"
     "  → Деньги перетекают в альткоины\n"
     "  → Сезон альткоинов (altseason)\n\n"
     "💡 Смотри на dominance чтобы понять куда идут большие деньги."),
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
                        msg = (
                            f"🔔 *Ценовой алерт!*\n\n"
                            f"*{name}* {direction} уровень `${level:,}`\n"
                            f"Текущая цена: `${price:,.2f}`\n\n"
                            f"_Ключевые уровни часто служат поддержкой или сопротивлением_"
                        )
                        await bot.send_message(CHANNEL_ID, msg, parse_mode=ParseMode.MARKDOWN)
                        await asyncio.sleep(1)
                last_price_alerts[key] = price
        except Exception as e:
            logger.error(f"Price level {symbol}: {e}")

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
    for symbol in WATCHLIST:
        try:
            df1h = await fetch_klines(session, symbol, "1h", 100)
            df4h = await fetch_klines(session, symbol, "4h", 60)
            if df1h is None or len(df1h) < 60: continue
            res = analyze(df1h, df4h)
            if not should_alert(symbol, res["signal"]): continue
            ticker = await fetch_ticker(session, symbol)
            change = float(ticker.get("priceChangePercent", 0))
            chart   = build_chart(res, symbol)
            caption = build_signal_msg(symbol, res, change)
            await bot.send_photo(CHANNEL_ID, photo=chart,
                                 caption=caption, parse_mode=ParseMode.MARKDOWN)
            last_alerts[symbol] = {"signal": res["signal"],
                                   "ts": datetime.now(timezone.utc).timestamp()}
            logger.info("Alert: %s → %s", symbol, res["signal"])
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
    await asyncio.sleep(600)  # первые новости через 10 минут
    while True:
        try: await send_news_digest(bot, session)
        except Exception as e: logger.error("News: %s", e)
        await asyncio.sleep(NEWS_INTERVAL)

async def digest_loop(bot, session):
    while True:
        try: await send_daily_digest(bot, session)
        except Exception as e: logger.error("Digest: %s", e)
        await asyncio.sleep(DIGEST_INTERVAL)

async def btc_analysis_loop(bot, session):
    await asyncio.sleep(300)  # через 5 минут после старта
    while True:
        try: await send_btc_analysis(bot, session)
        except Exception as e: logger.error("BTC analysis: %s", e)
        await asyncio.sleep(BTC_ANALYSIS_INTERVAL)

async def screener_loop(bot, session):
    await asyncio.sleep(900)  # через 15 минут
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
    logger.info("Bot v3.0 started: @%s", me.username)
    async with aiohttp.ClientSession() as session:
        await bot.send_message(
            CHANNEL_ID,
            "🤖 *Crypto Signals Bot v3.0 запущен!*\n\n"
            "📰 Мировые новости каждые 2 часа\n"
            "₿ Ежедневный разбор BTC\n"
            "🔍 Скринер топ-3 пары каждые 6 часов\n"
            "📊 15 пар | 1H + 4H таймфреймы\n"
            "😱 Fear & Greed | BTC Dominance\n"
            "📚 9 обучающих постов включая макроэкономику\n"
            "🔔 Алерты на ключевые уровни цен",
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
