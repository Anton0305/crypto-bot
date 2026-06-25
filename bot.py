"""
Crypto Signals Telegram Bot
Алерты в реальном времени + графики + новости
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import aiohttp
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import io

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHANNEL_ID     = os.getenv("CHANNEL_ID", "@your_channel_here")  # e.g. @CryptoSignalsXYZ

WATCHLIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ENAUSDT",
    # Extra interesting pairs
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT",  "SUIUSDT", "TIAUSDT",  "WIFUSDT", "PENDLEUSDT",
]

INTERVAL       = "1h"   # Candle interval for signals
CHECK_INTERVAL = 300    # Check every 5 minutes (seconds)
RSI_PERIOD     = 14
EMA_FAST       = 9
EMA_SLOW       = 21
EMA_TREND      = 50
VOLUME_MULT    = 1.5    # Volume spike threshold (x times average)

BINANCE_BASE   = "https://api.binance.com/api/v3"
NEWS_API       = "https://cryptopanic.com/api/v1/posts/?auth_token={token}&filter=hot&currencies={cur}"
CRYPTOPANIC_TOKEN = os.getenv("CRYPTOPANIC_TOKEN", "YOUR_CRYPTOPANIC_TOKEN")

# Track last alerts to avoid spam
last_alerts: dict[str, dict] = {}

# ── BINANCE DATA ──────────────────────────────────────────────────────────────

async def fetch_klines(session: aiohttp.ClientSession, symbol: str, interval: str = "1h", limit: int = 100):
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with session.get(url, params=params) as r:
        data = await r.json()
    if not isinstance(data, list) or len(data) < 2:
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


async def fetch_ticker(session: aiohttp.ClientSession, symbol: str):
    url = f"{BINANCE_BASE}/ticker/24hr"
    async with session.get(url, params={"symbol": symbol}) as r:
        return await r.json()


async def fetch_all_tickers(session: aiohttp.ClientSession):
    url = f"{BINANCE_BASE}/ticker/24hr"
    async with session.get(url) as r:
        return await r.json()

# ── INDICATORS ────────────────────────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist


def calc_bollinger(series: pd.Series, period: int = 20):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + 2*std, mid, mid - 2*std


def analyze(df: pd.DataFrame) -> dict:
    close = df["close"]
    vol   = df["volume"]

    rsi = calc_rsi(close, RSI_PERIOD)
    macd, macd_sig, macd_hist = calc_macd(close)
    ema_f = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_s = close.ewm(span=EMA_SLOW, adjust=False).mean()
    ema_t = close.ewm(span=EMA_TREND, adjust=False).mean()
    bb_up, bb_mid, bb_low = calc_bollinger(close)

    last_rsi       = rsi.iloc[-1]
    last_macd_hist = macd_hist.iloc[-1]
    prev_macd_hist = macd_hist.iloc[-2]
    last_close     = close.iloc[-1]
    last_vol       = vol.iloc[-1]
    avg_vol        = vol.rolling(20).mean().iloc[-1]

    vol_spike = last_vol > avg_vol * VOLUME_MULT

    # EMA alignment
    bullish_ema = ema_f.iloc[-1] > ema_s.iloc[-1] > ema_t.iloc[-1]
    bearish_ema = ema_f.iloc[-1] < ema_s.iloc[-1] < ema_t.iloc[-1]

    # MACD cross
    macd_bullish_cross = prev_macd_hist < 0 and last_macd_hist > 0
    macd_bearish_cross = prev_macd_hist > 0 and last_macd_hist < 0

    # Bollinger squeeze / breakout
    bb_squeeze  = (bb_up.iloc[-1] - bb_low.iloc[-1]) / bb_mid.iloc[-1] < 0.04
    bb_break_up = last_close > bb_up.iloc[-1]
    bb_break_dn = last_close < bb_low.iloc[-1]

    # Signal scoring
    buy_score  = 0
    sell_score = 0

    if last_rsi < 35:               buy_score  += 2
    elif last_rsi < 45:             buy_score  += 1
    if last_rsi > 65:               sell_score += 2
    elif last_rsi > 55:             sell_score += 1

    if macd_bullish_cross:          buy_score  += 3
    if macd_bearish_cross:          sell_score += 3
    if last_macd_hist > 0:          buy_score  += 1
    else:                           sell_score += 1

    if bullish_ema:                 buy_score  += 2
    if bearish_ema:                 sell_score += 2

    if bb_break_up and vol_spike:   buy_score  += 2
    if bb_break_dn and vol_spike:   sell_score += 2

    if vol_spike:
        if last_close > close.iloc[-2]: buy_score  += 1
        else:                           sell_score += 1

    # Determine signal
    if buy_score >= 5 and buy_score > sell_score + 2:
        if buy_score >= 8:   signal = "STRONG_BUY"
        else:                signal = "BUY"
    elif sell_score >= 5 and sell_score > buy_score + 2:
        if sell_score >= 8:  signal = "STRONG_SELL"
        else:                signal = "SELL"
    else:
        signal = "HOLD"

    # Simple TP/SL
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    tp1 = last_close + atr * 1.5
    tp2 = last_close + atr * 3.0
    sl  = last_close - atr * 1.0
    if "SELL" in signal:
        tp1 = last_close - atr * 1.5
        tp2 = last_close - atr * 3.0
        sl  = last_close + atr * 1.0

    return {
        "signal":     signal,
        "buy_score":  buy_score,
        "sell_score": sell_score,
        "price":      last_close,
        "rsi":        last_rsi,
        "macd_hist":  last_macd_hist,
        "ema_fast":   ema_f.iloc[-1],
        "ema_slow":   ema_s.iloc[-1],
        "ema_trend":  ema_t.iloc[-1],
        "bb_up":      bb_up.iloc[-1],
        "bb_low":     bb_low.iloc[-1],
        "vol_spike":  vol_spike,
        "tp1": tp1, "tp2": tp2, "sl": sl,
        "df": df,
        "rsi_series":      rsi,
        "macd_series":     macd,
        "macd_sig_series": macd_sig,
        "macd_hist_series":macd_hist,
        "ema_f_series":    ema_f,
        "ema_s_series":    ema_s,
        "ema_t_series":    ema_t,
        "bb_up_series":    bb_up,
        "bb_mid_series":   bb_mid,
        "bb_low_series":   bb_low,
    }

# ── CHART ─────────────────────────────────────────────────────────────────────

def build_chart(res: dict, symbol: str) -> bytes:
    df  = res["df"].copy().tail(60)
    signal = res["signal"]

    color_map = {
        "STRONG_BUY":  "#00ff88",
        "BUY":         "#00cc66",
        "HOLD":        "#ffcc00",
        "SELL":        "#ff6644",
        "STRONG_SELL": "#ff2200",
    }
    signal_color = color_map.get(signal, "#aaaaaa")

    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    gs = GridSpec(4, 1, figure=fig, hspace=0.08,
                  height_ratios=[3, 0.8, 0.8, 0.8])

    ax1 = fig.add_subplot(gs[0])  # Candles + BBands + EMAs
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Volume
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # MACD
    ax4 = fig.add_subplot(gs[3], sharex=ax1)  # RSI

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_color("#21262d")
        ax.yaxis.label.set_color("#8b949e")

    x = np.arange(len(df))
    # Candles
    for i, (_, row) in enumerate(df.iterrows()):
        color = "#26a641" if row["close"] >= row["open"] else "#f85149"
        ax1.plot([i, i], [row["low"], row["high"]], color=color, lw=0.8)
        ax1.bar(i, abs(row["close"]-row["open"]), bottom=min(row["open"],row["close"]),
                color=color, width=0.7, alpha=0.9)

    # Bollinger Bands
    ax1.fill_between(x, res["bb_up_series"].values[-60:], res["bb_low_series"].values[-60:],
                     alpha=0.08, color="#58a6ff")
    ax1.plot(x, res["bb_up_series"].values[-60:],  color="#58a6ff", lw=0.7, alpha=0.5)
    ax1.plot(x, res["bb_mid_series"].values[-60:], color="#58a6ff", lw=0.7, alpha=0.3, linestyle="--")
    ax1.plot(x, res["bb_low_series"].values[-60:], color="#58a6ff", lw=0.7, alpha=0.5)

    # EMAs
    ax1.plot(x, res["ema_f_series"].values[-60:], color="#f0c419", lw=1.2, label=f"EMA{EMA_FAST}")
    ax1.plot(x, res["ema_s_series"].values[-60:], color="#e36bdf", lw=1.2, label=f"EMA{EMA_SLOW}")
    ax1.plot(x, res["ema_t_series"].values[-60:], color="#58a6ff", lw=1.2, label=f"EMA{EMA_TREND}")

    # TP / SL lines
    if signal != "HOLD":
        ax1.axhline(res["tp1"], color="#26a641", lw=0.8, linestyle="--", alpha=0.7)
        ax1.axhline(res["tp2"], color="#26a641", lw=0.8, linestyle=":",  alpha=0.5)
        ax1.axhline(res["sl"],  color="#f85149", lw=0.8, linestyle="--", alpha=0.7)
        ax1.text(len(x)-1, res["tp1"], f" TP1", color="#26a641", va="center", fontsize=7)
        ax1.text(len(x)-1, res["tp2"], f" TP2", color="#26a641", va="center", fontsize=7)
        ax1.text(len(x)-1, res["sl"],  f" SL",  color="#f85149", va="center", fontsize=7)

    # Title
    price_str = f"${res['price']:,.4f}" if res['price'] < 1 else f"${res['price']:,.2f}"
    ax1.set_title(
        f"{symbol}  {price_str}   [{signal}]",
        color=signal_color, fontsize=14, fontweight="bold", pad=8,
        fontfamily="monospace"
    )
    leg = ax1.legend(fontsize=7, loc="upper left",
                     facecolor="#161b22", edgecolor="#21262d", labelcolor="#8b949e")

    # Volume
    vol_colors = ["#26a641" if df["close"].iloc[i] >= df["open"].iloc[i] else "#f85149"
                  for i in range(len(df))]
    ax2.bar(x, df["volume"].values, color=vol_colors, alpha=0.7, width=0.7)
    avg_v = df["volume"].rolling(20).mean().values
    ax2.plot(x, avg_v, color="#f0c419", lw=0.8)
    ax2.set_ylabel("Volume", fontsize=7)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v>=1e6 else f"{v/1e3:.0f}K"))

    # MACD
    hist = res["macd_hist_series"].values[-60:]
    macd_v = res["macd_series"].values[-60:]
    msig_v = res["macd_sig_series"].values[-60:]
    bar_colors = ["#26a641" if h >= 0 else "#f85149" for h in hist]
    ax3.bar(x, hist, color=bar_colors, alpha=0.7, width=0.7)
    ax3.plot(x, macd_v,  color="#58a6ff", lw=1.0)
    ax3.plot(x, msig_v, color="#f0c419", lw=1.0)
    ax3.axhline(0, color="#21262d", lw=0.8)
    ax3.set_ylabel("MACD", fontsize=7)

    # RSI
    rsi_v = res["rsi_series"].values[-60:]
    ax4.plot(x, rsi_v, color="#e36bdf", lw=1.2)
    ax4.axhline(70, color="#f85149", lw=0.7, linestyle="--", alpha=0.6)
    ax4.axhline(30, color="#26a641", lw=0.7, linestyle="--", alpha=0.6)
    ax4.axhline(50, color="#21262d", lw=0.6)
    ax4.fill_between(x, rsi_v, 70, where=(rsi_v>=70), alpha=0.15, color="#f85149")
    ax4.fill_between(x, rsi_v, 30, where=(rsi_v<=30), alpha=0.15, color="#26a641")
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("RSI", fontsize=7)
    ax4.text(len(x)-1, rsi_v[-1], f"  {rsi_v[-1]:.1f}", color="#e36bdf", va="center", fontsize=7)

    # X-axis labels (every 10 candles)
    tick_pos = list(range(0, len(df), 10))
    tick_lab = [df.index[i].strftime("%d/%m %H:%M") for i in tick_pos]
    ax4.set_xticks(tick_pos)
    ax4.set_xticklabels(tick_lab, rotation=30, ha="right", fontsize=7, color="#8b949e")
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── MESSAGE FORMATTING ────────────────────────────────────────────────────────

SIGNAL_EMOJI = {
    "STRONG_BUY":  "🚀🟢",
    "BUY":         "🟢",
    "HOLD":        "⏸️",
    "SELL":        "🔴",
    "STRONG_SELL": "🔴💀",
}

def format_price(price: float) -> str:
    if price >= 1000:  return f"${price:,.2f}"
    if price >= 1:     return f"${price:.4f}"
    return f"${price:.6f}"


def build_signal_message(symbol: str, res: dict, change_24h: float) -> str:
    s      = res["signal"]
    emoji  = SIGNAL_EMOJI.get(s, "")
    p      = res["price"]
    ch_str = f"{'🔺' if change_24h>=0 else '🔻'}{abs(change_24h):.2f}%"
    now    = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    name   = symbol.replace("USDT","")

    lines = [
        f"{emoji} *{name}/USDT — {s}*",
        f"",
        f"💵 Цена: `{format_price(p)}`   {ch_str}",
        f"📅 `{now}`",
        f"",
        f"📊 *Технический анализ:*",
        f"  RSI({RSI_PERIOD}): `{res['rsi']:.1f}`"
        + (" 🔥 Перекуплен" if res['rsi']>70 else " 🧊 Перепродан" if res['rsi']<30 else ""),
        f"  MACD Hist: `{res['macd_hist']:+.4f}`",
        f"  EMA{EMA_FAST}: `{format_price(res['ema_fast'])}`",
        f"  EMA{EMA_SLOW}: `{format_price(res['ema_slow'])}`",
        f"  EMA{EMA_TREND}: `{format_price(res['ema_trend'])}`",
        f"  BB: `{format_price(res['bb_low'])}` — `{format_price(res['bb_up'])}`",
    ]
    if res["vol_spike"]:
        lines.append(f"  📈 *Всплеск объёма!*")

    if s != "HOLD":
        action = "покупки" if "BUY" in s else "продажи"
        lines += [
            f"",
            f"🎯 *Уровни для {action}:*",
            f"  🟢 TP1: `{format_price(res['tp1'])}`",
            f"  🟢 TP2: `{format_price(res['tp2'])}`",
            f"  🛑 SL:  `{format_price(res['sl'])}`",
        ]

    lines += [
        f"",
        f"⚡ Сигнал: *{res['buy_score']}🟢 / {res['sell_score']}🔴*",
        f"",
        f"_Интервал: {INTERVAL} | Только для информации, не является финансовым советом_",
    ]
    return "\n".join(lines)

# ── NEWS ──────────────────────────────────────────────────────────────────────

async def fetch_news(session: aiohttp.ClientSession, currency: str = "BTC,ETH,SOL") -> list[dict]:
    if CRYPTOPANIC_TOKEN == "YOUR_CRYPTOPANIC_TOKEN":
        return []
    url = NEWS_API.format(token=CRYPTOPANIC_TOKEN, cur=currency)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            return data.get("results", [])[:5]
    except Exception:
        return []


def build_news_message(news: list[dict]) -> str:
    if not news:
        return ""
    lines = ["📰 *Горячие новости крипторынка:*", ""]
    for item in news:
        title = item.get("title", "")
        url   = item.get("url", "")
        currencies = ", ".join(c["code"] for c in item.get("currencies", []))
        votes  = item.get("votes", {})
        pos    = votes.get("positive", 0)
        neg    = votes.get("negative", 0)
        mood   = "🟢" if pos > neg else "🔴" if neg > pos else "⚪"
        lines.append(f"{mood} [{title}]({url})")
        if currencies:
            lines.append(f"   _#{currencies}_")
        lines.append("")
    return "\n".join(lines)

# ── DAILY DIGEST ──────────────────────────────────────────────────────────────

async def send_daily_digest(bot: Bot, session: aiohttp.ClientSession):
    tickers = await fetch_all_tickers(session)
    usdt_pairs = {t["symbol"]: t for t in tickers
                  if t["symbol"] in WATCHLIST}

    rows = []
    for sym in WATCHLIST:
        t = usdt_pairs.get(sym)
        if not t:
            continue
        p   = float(t["lastPrice"])
        ch  = float(t["priceChangePercent"])
        vol = float(t["quoteVolume"])
        rows.append((sym.replace("USDT",""), p, ch, vol))

    rows.sort(key=lambda r: abs(r[2]), reverse=True)

    lines = [
        "🌅 *УТРЕННИЙ ДАЙДЖЕСТ*",
        f"_{datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}_",
        "",
        "```",
        f"{'Пара':<8} {'Цена':>12} {'24h':>8} {'Объём 24h':>14}",
        "─"*46,
    ]
    for name, price, ch, vol in rows:
        arrow = "▲" if ch >= 0 else "▼"
        p_str = f"${price:>10,.2f}" if price>=1 else f"${price:>10.4f}"
        lines.append(f"{name:<8} {p_str} {arrow}{abs(ch):>5.1f}% ${vol/1e6:>9.1f}M")
    lines.append("```")

    # Top gainer/loser
    gainer = max(rows, key=lambda r: r[2])
    loser  = min(rows, key=lambda r: r[2])
    lines += [
        "",
        f"🏆 Лидер роста:   *{gainer[0]}* `+{gainer[2]:.2f}%`",
        f"📉 Лидер падения: *{loser[0]}*  `{loser[2]:.2f}%`",
        "",
        "_Следующий дайджест через 24 часа_",
    ]

    await bot.send_message(CHANNEL_ID, "\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)

# ── SIGNAL LOOP ───────────────────────────────────────────────────────────────

def should_alert(symbol: str, new_signal: str) -> bool:
    """Only alert on new signals or if signal changed."""
    if new_signal == "HOLD":
        return False
    prev = last_alerts.get(symbol, {})
    if prev.get("signal") == new_signal:
        # Don't repeat same signal for 4 hours
        ts = prev.get("ts", 0)
        if (datetime.now(timezone.utc).timestamp() - ts) < 4 * 3600:
            return False
    return True


async def check_signals(bot: Bot, session: aiohttp.ClientSession):
    logger.info("Checking signals for %d pairs…", len(WATCHLIST))
    for symbol in WATCHLIST:
        try:
            df = await fetch_klines(session, symbol, INTERVAL, limit=100)
            if df is None or len(df) < 60:
                continue

            res = analyze(df)
            sig = res["signal"]

            if not should_alert(symbol, sig):
                continue

            ticker = await fetch_ticker(session, symbol)
            change_24h = float(ticker.get("priceChangePercent", 0))

            # Build and send chart
            chart_bytes = build_chart(res, symbol)
            caption     = build_signal_message(symbol, res, change_24h)

            await bot.send_photo(
                CHANNEL_ID,
                photo=chart_bytes,
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
            )

            last_alerts[symbol] = {
                "signal": sig,
                "ts": datetime.now(timezone.utc).timestamp(),
            }
            logger.info("Alert sent: %s → %s", symbol, sig)

            await asyncio.sleep(1)  # avoid Telegram rate limits

        except TelegramError as e:
            logger.error("Telegram error for %s: %s", symbol, e)
        except Exception as e:
            logger.exception("Error processing %s: %s", symbol, e)


async def news_loop(bot: Bot, session: aiohttp.ClientSession):
    """Send news digest every 6 hours."""
    while True:
        try:
            news = await fetch_news(session)
            msg  = build_news_message(news)
            if msg:
                await bot.send_message(CHANNEL_ID, msg,
                                       parse_mode=ParseMode.MARKDOWN,
                                       disable_web_page_preview=True)
        except Exception as e:
            logger.error("News error: %s", e)
        await asyncio.sleep(6 * 3600)


async def digest_loop(bot: Bot, session: aiohttp.ClientSession):
    """Send daily market digest every 24 hours."""
    while True:
        try:
            await send_daily_digest(bot, session)
        except Exception as e:
            logger.error("Digest error: %s", e)
        await asyncio.sleep(24 * 3600)


async def signal_loop(bot: Bot, session: aiohttp.ClientSession):
    """Main signal checking loop."""
    while True:
        await check_signals(bot, session)
        await asyncio.sleep(CHECK_INTERVAL)


async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    me  = await bot.get_me()
    logger.info("Bot started: @%s", me.username)

    async with aiohttp.ClientSession() as session:
        # Startup message
        await bot.send_message(
            CHANNEL_ID,
            "🤖 *Crypto Signals Bot запущен!*\n\n"
            f"📊 Отслеживаю {len(WATCHLIST)} пар\n"
            f"⏱ Интервал проверки: каждые {CHECK_INTERVAL//60} мин\n"
            f"📈 Сигналы на основе: RSI, MACD, EMA, Bollinger Bands\n"
            f"🔔 Алерты при смене сигнала",
            parse_mode=ParseMode.MARKDOWN,
        )
        await asyncio.gather(
            signal_loop(bot, session),
            news_loop(bot, session),
            digest_loop(bot, session),
        )


if __name__ == "__main__":
    asyncio.run(main())
