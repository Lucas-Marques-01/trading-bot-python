# ============================================================
#  bot_analise_tecnica.py — SHARED MODULE (CONSERVATIVE + AGGRESSIVE)
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import sqlite3
from datetime import datetime
import pytz
import math

# ================= LOGGING ==================
from logger_bot import LoggerHandler, registrar_evento
handler = LoggerHandler()
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# ================= CONFIG GLOBAL ==================
ACTIONS = ["PETR4.SA", "PETR3.SA", "VALE3.SA", "BBAS3.SA", "ITUB4.SA",
           "BBDC4.SA", "BPAC11.SA", "SUZB3.SA", "LREN3.SA", "WEGE3.SA", "ABEV3.SA"]

TIMEFRAMES = {
    "1d": "1d",
    "1w": "1wk",
    "4h": "4h"
}

SAO_PAULO_TZ = pytz.timezone("America/Sao_Paulo")


def criar_tabela_ml_predictions():
    conn = sqlite3.connect("alerts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            timestamp TEXT,
            predicted_type TEXT,
            confidence REAL
        );
    """)
    conn.commit()
    conn.close()

criar_tabela_ml_predictions()


# ================= DATABASE =================
conn = sqlite3.connect("alerts.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    timeframe TEXT,
    type TEXT,
    price REAL,
    reason TEXT,
    timestamp TEXT,
    atr REAL,
    rsi REAL,
    macd_hist REAL,
    tendencia_base TEXT,
    volume_rel REAL,
    rr REAL,
    outcome REAL
)
""")
conn.commit()


# ============================================================
# 1) DATA LOADING
# ============================================================
def carregar_dados(ticker, intervalo):
    try:
        if intervalo.lower() in ["1d"]:
            period = "6mo"
        elif intervalo.lower() in ["1wk", "1w"]:
            period = "2y"
        elif intervalo.lower() in ["4h"]:
            period = "60d"  
        else:
            period = "6mo"

        df = yf.download(ticker, period=period, interval=intervalo, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Adj Close" in df.columns and "Close" not in df.columns:
            df["Close"] = df["Adj Close"]

        
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan

        df.dropna(how='all', inplace=True)
        if df.empty or df["Volume"].isna().all() or df["Volume"].sum() == 0:
            return None

        return df
    except Exception as e:
        logging.error(f"Erro ao baixar {ticker} intervalo {intervalo}: {e}")
        return None


# ============================================================
# 2) TECHNICAL INDICATORS
# ============================================================
def calcular_indicadores(df):
    df = df.copy()
    if df is None or df.empty:
        return df

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    
    df["MM9"] = close.rolling(9).mean()
    df["MM20"] = close.rolling(20).mean()
    df["MM50"] = close.rolling(50).mean()

    
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    
    df["CUM_VOL"] = df["Volume"].cumsum()
    df["CUM_PV"] = (df["Close"] * df["Volume"]).cumsum()
    df["VWAP"] = df["CUM_PV"] / df["CUM_VOL"].replace(0, np.nan)

    df["VOL_MA"] = df["Volume"].rolling(window=20, min_periods=5).mean()
    df["VOL_REL"] = df["Volume"] / df["VOL_MA"].replace(0, np.nan)

    return df


# ============================================================
# 3) CANDLESTICK PATTERNS (CONFIRMATION)
# ============================================================
def detectar_padrao_candle(df):
    if df is None or len(df) < 2:
        return None, None

    try:
        o1, h1, l1, c1 = df["Open"].iloc[-2], df["High"].iloc[-2], df["Low"].iloc[-2], df["Close"].iloc[-2]
        o2, h2, l2, c2 = df["Open"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
    except Exception:
        return None, None

    corpo = abs(c2 - o2)
    sombra_sup = h2 - max(c2, o2)
    sombra_inf = min(c2, o2) - l2

    if corpo < (sombra_inf * 0.5) and sombra_inf > 2 * corpo:
        return "COMPRA", "HAMMER"
    if corpo < (sombra_sup * 0.5) and sombra_sup > 2 * corpo:
        return "VENDA", "SHOOTING_STAR"
    if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
        return "COMPRA", "ENGULFING"
    if c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
        return "VENDA", "BEAR_ENGULFING"
    if (h2 - l2) > 0 and abs(c2 - o2) / (h2 - l2 + 1e-9) > 0.9:
        return ("COMPRA", "MARUBOZU") if c2 > o2 else ("VENDA", "MARUBOZU_BAIXA")

    return None, None


# ============================================================
# 4) EXTRACTING VALUES
# ============================================================
def extrair_valores(df):
    def last(col):
        try:
            return float(df[col].iloc[-1])
        except Exception:
            return None

    return {
        "close": last("Close"),
        "mm20": last("MM20"),
        "mm50": last("MM50"),
        "rsi": last("RSI"),
        "macd_hist": last("MACD_HIST"),
        "macd_hist_prev": df["MACD_HIST"].iloc[-2] if len(df) > 2 else None,
        "atr": last("ATR"),
        "volume": last("Volume"),
        "volume_medio": df["Volume"].tail(60).mean(),
        "volume_rel": last("VOL_REL"),
        "vwap": last("VWAP"),
    }


# ============================================================
# ---------- START: CONSERVATIVE V2 INTEGRATED FUNCTIONS ----------
# ============================================================
# Helpers: pivots, pullbacks, breakouts, slope, trendline, volatility, fib

def detect_swings(df, lookback=3):

    highs = []
    lows = []
    n = len(df)
    for i in range(lookback, n - lookback):
        window_high = df['High'].iloc[i - lookback:i + lookback + 1]
        window_low = df['Low'].iloc[i - lookback:i + lookback + 1]
        if df['High'].iloc[i] == window_high.max():
            highs.append(i)
        if df['Low'].iloc[i] == window_low.min():
            lows.append(i)
    return highs, lows


def detect_breakout(df, window=20):

    res = {"breakout_high": False, "breakout_low": False, "dist_pct": None, "level": None}
    if len(df) < window + 2:
        return res

    close = df['Close'].iloc[-1]
    prev_high = df['High'].iloc[-(window+1):-1].max()
    prev_low = df['Low'].iloc[-(window+1):-1].min()

    if prev_high and close > prev_high:
        res['breakout_high'] = True
        res['dist_pct'] = (close - prev_high) / prev_high * 100
        res['level'] = prev_high
    if prev_low and close < prev_low:
        res['breakout_low'] = True
        res['dist_pct'] = (prev_low - close) / prev_low * 100
        res['level'] = prev_low

    return res


def fib_retracement_levels(high, low):

    diff = high - low
    return {
        "38.2": high - 0.382 * diff,
        "50.0": high - 0.5 * diff,
        "61.8": high - 0.618 * diff
    }


def detect_pullback(df, recent_high=None, recent_low=None, max_retrace=0.618):

    res = {"is_pullback": False, "retrace_pct": None, "fib_hit": None, "to_mm20": False, "to_mm50": False}
    if df is None or len(df) < 10:
        return res

    close = df['Close'].iloc[-1]
    if recent_high is None:
        recent_high = df['High'].iloc[-21:-1].max()  # busca topo nos últimos 20 candles
    if recent_low is None:
        recent_low = df['Low'].iloc[-21:-1].min()

    if recent_high is None or recent_low is None or recent_high == recent_low:
        return res

    diff = recent_high - recent_low
    retrace = (recent_high - close) / (diff + 1e-9)  # 0..1
    res['retrace_pct'] = retrace

    
    levels = fib_retracement_levels(recent_high, recent_low)
    for k, v in levels.items():
        
        if abs(close - v) / (v + 1e-9) <= 0.005:
            res['fib_hit'] = k

    
    if 'MM20' in df.columns:
        mm20 = df['MM20'].iloc[-1]
        if not np.isnan(mm20) and abs(close - mm20) / (mm20 + 1e-9) <= 0.03:  # dentro de 3%
            res['to_mm20'] = True
    if 'MM50' in df.columns:
        mm50 = df['MM50'].iloc[-1]
        if not np.isnan(mm50) and abs(close - mm50) / (mm50 + 1e-9) <= 0.03:  # dentro de 3%
            res['to_mm50'] = True

    
    if retrace <= max_retrace and (res['fib_hit'] is not None or res['to_mm20'] or res['to_mm50']):
        res['is_pullback'] = True

    return res


def trend_slope(series, window=30):
    """
    Calcula slope normalizada por média usando polyfit (sem dependências extras).
    Retorna slope normalizada por preço médio.
    """
    if series is None or len(series) < window:
        return None
    y = np.array(series[-window:], dtype=float)
    x = np.arange(len(y))
    
    try:
        m, b = np.polyfit(x, y, 1)
        mean_price = np.mean(y) if np.mean(y) != 0 else 1.0
        return (m / mean_price)
    except Exception:
        return None


def trendline_from_points(df, kind='support', window=30):

    if df is None or len(df) < window:
        return None, None
    if kind == 'support':
        pts = df['Low'].tail(window).values.astype(float)
    else:
        pts = df['High'].tail(window).values.astype(float)
    x = np.arange(len(pts))
    try:
        m, b = np.polyfit(x, pts, 1)
        last_on_line = b + m * (len(pts) - 1)
        mean_price = np.mean(pts) if np.mean(pts) != 0 else 1.0
        return (m / mean_price), last_on_line
    except Exception:
        return None, None


def is_volatility_increasing(df, short_window=10, long_window=40):

    if df is None or 'ATR' not in df.columns:
        return False, None
    atr = df['ATR'].dropna()
    if len(atr) < long_window:
        return False, None
    short = atr.iloc[-short_window:].mean()
    long = atr.iloc[-long_window:].mean()
    return (short > long), {"atr_short": short, "atr_long": long}


def consecutive_direction(df, lookback=10):

    if df is None or len(df) < 2:
        return 0, 0
    closes = df['Close'].iloc[-lookback:].values
    up = down = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            up += 1
            down = 0
        elif closes[i] < closes[i-1]:
            down += 1
            up = 0
        else:
            up = down = 0
    return up, down


def structure_direction_from_swings(df, swing_highs, swing_lows):

    try:
        last_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        last_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        if len(last_highs) >= 2 and len(last_lows) >= 2:
            # pegar valores dos pivôs
            highs_vals = [df['High'].iloc[i] for i in last_highs]
            lows_vals = [df['Low'].iloc[i] for i in last_lows]
            # se highs subindo e lows subindo -> bull
            if highs_vals[-1] > highs_vals[0] and lows_vals[-1] > lows_vals[0]:
                return 'BULL'
            if highs_vals[-1] < highs_vals[0] and lows_vals[-1] < lows_vals[0]:
                return 'BEAR'
    except Exception:
        pass
    return 'SIDE'


# ============================================================
# 5) CONSERVATIVE RULES (LEGACY)
# ============================================================
def regras_conservadoras(df, v, ticker):
    BLUE_CHIPS = ["PETR4","PETR3","VALE3","BBAS3","ITUB4","BBDC4","BPAC11","SUZB3","LREN3","WEGE3","ABEV3"]
    base = ticker.replace(".SA", "")

    if v["volume"] < v["volume_medio"] * 0.5 and base not in BLUE_CHIPS:
        return None, "Volume baixo — evitar operar (conservador)"

    if v["atr"] < (df["Close"].tail(60).mean() * 0.003):
        return None, "ATR baixo — mercado lateral (conservador)"

    tipo, nome = detectar_padrao_candle(df)

    if tipo == "COMPRA":
        if v["close"] > v["mm20"] > v["mm50"] and v["macd_hist"] > 0 and v["macd_hist"] > v["macd_hist_prev"] and v["rsi"] < 70:
            return "COMPRA", f"Padrão {nome} + tendência (20>50) + MACD crescente + volume ok"
        else:
            return None, "COMPRA descartada — sem confirmação"

    if tipo == "VENDA":
        if v["close"] < v["mm20"] < v["mm50"] and v["macd_hist"] < 0 and v["macd_hist"] < v["macd_hist_prev"] and v["rsi"] > 30:
            return "VENDA", f"Padrão {nome} + tendência (20<50) + MACD decrescente + volume ok"
        else:
            return None, "VENDA descartada — sem confirmação"

    return None, "Sem alinhamento técnico conservador"


# ============================================================
# 5) CONSERVATIVE RULES V2 (BUY/SELL WITH STRICTER RULES)
# ============================================================
def regras_conservadoras_v2(df, v, ticker,
                            min_candles=60,
                            breakout_window=20,
                            pullback_max_retrace=0.618,
                            require_confluence=2):


    base = ticker.replace(".SA", "")

    
    if df is None or len(df) < min_candles:
        return None, f"Dados insuficientes (min {min_candles} candles)"


    if v.get("volume") is None or v.get("volume_medio") is None:
        return None, "Volume ausente"
    if v["volume"] < max(1.0, v["volume_medio"] * 0.5) and base not in [s.replace(".SA", "") for s in ACTIONS]:
        return None, "Volume baixo — evitar operar (conservador V2)"

    avg_close = df['Close'].tail(60).mean()
    if v.get("atr") is None or v["atr"] < (avg_close * 0.0025):
        return None, "ATR baixo — mercado lateral (conservador V2)"

    
    swing_highs, swing_lows = detect_swings(df, lookback=3)
    structure = structure_direction_from_swings(df, swing_highs, swing_lows)

    
    breakout = detect_breakout(df, window=breakout_window)

    
    recent_high_idx = swing_highs[-1] if swing_highs else None
    recent_low_idx = swing_lows[-1] if swing_lows else None
    recent_high = df['High'].iloc[recent_high_idx] if recent_high_idx is not None else None
    recent_low = df['Low'].iloc[recent_low_idx] if recent_low_idx is not None else None
    pb = detect_pullback(df, recent_high=recent_high, recent_low=recent_low, max_retrace=pullback_max_retrace)

    
    slope_mm20 = trend_slope(df['MM20'].values if 'MM20' in df.columns else df['Close'].rolling(20).mean().values, window=30)
    lta_slope, lta_value = trendline_from_points(df, kind='support', window=30)
    ltb_slope, ltb_value = trendline_from_points(df, kind='resistance', window=30)
    vol_inc, vol_metrics = is_volatility_increasing(df, short_window=10, long_window=40)
    up_count, down_count = consecutive_direction(df, lookback=10)

    
    tipo_padrao, nome_padrao = detectar_padrao_candle(df)

    
    buy_signals = []
    sell_signals = []

    
    macd_hist = v.get("macd_hist")
    rsi = v.get("rsi")
    mm20 = v.get("mm20")
    mm50 = v.get("mm50")
    close = v.get("close")

    # ---------- BUY SIGNALS ----------
    if structure == 'BULL' and pb.get('is_pullback') and (mm20 is not None and mm50 is not None and mm20 > mm50):
        buy_signals.append("Structure(BULL)+Pullback")

    if breakout.get('breakout_high') and v.get("volume_rel") and v.get("volume_rel") > 1.0 and vol_inc:
        if breakout.get('dist_pct') and breakout['dist_pct'] > 0.3:
            buy_signals.append("BreakoutHigh+Vol+ATRup")

    if swing_lows:
        last_pivot = swing_lows[-1]
        up_after = 0
        for i in range(last_pivot + 1, min(last_pivot + 6, len(df))):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                up_after += 1
        if up_after >= 3 and mm20 is not None and mm50 is not None and mm20 > mm50:
            buy_signals.append("PivotLow+3up+MM20>MM50")

    if macd_hist is not None and macd_hist > 0 and (rsi is None or rsi < 70):
        buy_signals.append("MomentumPos")

    if tipo_padrao == "COMPRA":
        buy_signals.append(f"Padrão:{nome_padrao}")

    # ---------- SELL SIGNALS ----------
    # 1) bear structure + valid pullback for a short entry (e.g., pullback to the 20 MA before the breakdown)
    if structure == 'BEAR' and pb.get('is_pullback') and (mm20 is not None and mm50 is not None and mm20 < mm50):
        sell_signals.append("Structure(BEAR)+Pullback")

    # 2) downside breakout confirmed by volume/ATR
    if breakout.get('breakout_low') and v.get("volume_rel") and v.get("volume_rel") > 1.0 and vol_inc:
        if breakout.get('dist_pct') and breakout['dist_pct'] > 0.3:
            sell_signals.append("BreakoutLow+Vol+ATRup")

    # 3) confirmed high pivot + bearish confirmation candles
    if swing_highs:
        last_pivot_h = swing_highs[-1]
        down_after = 0
        for i in range(last_pivot_h + 1, min(last_pivot_h + 6, len(df))):
            if df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                down_after += 1
        if down_after >= 3 and mm20 is not None and mm50 is not None and mm20 < mm50:
            sell_signals.append("PivotHigh+3down+MM20<MM50")

    if macd_hist is not None and macd_hist < 0 and (rsi is None or rsi > 30):
        sell_signals.append("MomentumNeg")

    if tipo_padrao == "VENDA":
        sell_signals.append(f"Padrão:{nome_padrao}")

    # ---------- Safety rules ----------
    
    if slope_mm20 is not None:
        if slope_mm20 < -0.0005 and len(buy_signals) > 0:
            return None, "MM20 com slope fortemente negativo — evita compra"


    
    if len(buy_signals) > 0 and lta_slope is not None and lta_slope < 0:
        return None, "LTA descendente — evita compra"
    if len(sell_signals) > 0 and ltb_slope is not None and ltb_slope > 0:
        return None, "LTB ascendente — evita venda"

    # ---------- 11) decidir sinal com base em confluência (compras) ----------
    if len(buy_signals) >= require_confluence:
        motivo = " + ".join(buy_signals)
        if pb.get('fib_hit'):
            motivo += f" | Fib:{pb.get('fib_hit')}"
        return "COMPRA", motivo


    sell_req = require_confluence
    need_weekly_bear = False
    strong_breakout_low = False


    if breakout.get('breakout_low') and v.get("volume_rel") and v.get("volume_rel") > 1.5 and vol_inc:
        strong_breakout_low = True

    
    if slope_mm20 is not None and slope_mm20 > 0.0005:
        sell_req = require_confluence + 1
        
        if structure != 'BEAR' and not strong_breakout_low:
            
            return None, "MM20 com slope fortemente positivo — exige estrutura BEAR ou breakout baixo forte para vender"

    
    if len(sell_signals) >= sell_req:
        motivo = " + ".join(sell_signals)
        if pb.get('fib_hit'):
            motivo += f" | Fib:{pb.get('fib_hit')}"
        
        if v.get("volume_rel") is None or v.get("volume_rel") < 0.6:
            return None, "Volume relativo baixo — evita venda conservadora"
        return "VENDA", motivo

    
    if breakout.get('breakout_high') and v.get("volume_rel") and v.get("volume_rel") > 1.5 and vol_inc:
        return "COMPRA", "Breakout forte + volume_rel>1.5 + ATR crescente"
    if breakout.get('breakout_low') and v.get("volume_rel") and v.get("volume_rel") > 1.8 and vol_inc:
        
        return "VENDA", "Breakout baixo muito forte + volume_rel>1.8 + ATR crescente"

    return None, "Sem confluência técnica suficiente (conservador V2)"




import sqlite3

def obter_sinal_ml_do_banco(ticker):
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT predicted_type FROM ml_predictions
        WHERE ticker = ? ORDER BY id DESC LIMIT 1
    """, (ticker,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

def analisar_ticker_base(ticker, timeframe="1d"):
    intervalo = TIMEFRAMES.get(timeframe, timeframe)
    df = carregar_dados(ticker, intervalo)
    if df is None or len(df) < 30:
        return {
            "ticker": ticker,
            "close": None,
            "ema9": None,
            "high": None,
            "low": None,
            "prev_high": None,
            "prev_low": None,
            "tendencia_base": None,
            "volume_rel": None,
            "atr": None,
            "macd_hist": None,
            "rsi": None,              
            "rr": None,
            "rr_adjusted": None,
            "stops": None,
            "confirm_4h": False,
        }

    
    df = calcular_indicadores(df)

    close = df["Close"].iloc[-1]
    ema9 = df["MM9"].iloc[-1] if "MM9" in df.columns else None

    high = df["High"].iloc[-1]
    low = df["Low"].iloc[-1]

    prev_high = df["High"].iloc[-2] if len(df) > 1 else None
    prev_low = df["Low"].iloc[-2] if len(df) > 1 else None

    macd_hist = df["MACD_HIST"].iloc[-1] if "MACD_HIST" in df.columns else None
    atr = df["ATR"].iloc[-1] if "ATR" in df.columns else None
    volume_rel = df["VOL_REL"].iloc[-1] if "VOL_REL" in df.columns else None

    
    rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else None


    
    mm20 = df["MM20"].iloc[-1] if "MM20" in df.columns else None
    mm50 = df["MM50"].iloc[-1] if "MM50" in df.columns else None

    if mm20 is not None and mm50 is not None:
        if mm20 > mm50:
            tendencia = "UP"
        elif mm20 < mm50:
            tendencia = "DOWN"
        else:
            tendencia = "SIDE"
    else:
        tendencia = None

    return {
        "ticker": ticker,
        "close": close,
        "ema9": ema9,
        "high": high,
        "low": low,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "tendencia_base": tendencia,
        "volume_rel": volume_rel,
        "atr": atr,
        "macd_hist": macd_hist,
        "rsi": rsi,                
        "rr": None,
        "rr_adjusted": None,
        "stops": None,
        "confirm_4h": False,
    }



def analisar_ticker_agressivo(ticker, timeframe="1d", usar_ml=False):


    rich = analisar_ticker_base(ticker, timeframe=timeframe)

    preco = rich.get("close")
    rsi = rich.get("rsi")
    macd = rich.get("macd_hist")
    volume_rel = rich.get("volume_rel")
    tendencia = rich.get("tendencia_base")
    ema9 = rich.get("ema9")
    high = rich.get("high")
    low = rich.get("low")
    prev_high = rich.get("prev_high")
    prev_low = rich.get("prev_low")


    conf4h = check_4h_confirmation(ticker)


    if volume_rel is None:
        return montar_saida_agressiva(ticker, preco, "NEUTRO", "Volume ausente", rich)

    if volume_rel < 0.10:
        return montar_saida_agressiva(ticker, preco, "NEUTRO", "Volume fraco (<0.10)", rich)


    if rsi is None:
        return montar_saida_agressiva(ticker, preco, "NEUTRO", "RSI insuficiente", rich)


    if rsi > 75:
        return montar_saida_agressiva(ticker, preco, "NEUTRO", "RSI muito esticado (>75)", rich)


    if rsi < 25:
        return montar_saida_agressiva(ticker, preco, "NEUTRO", "RSI muito sobrevendido (<25)", rich)


    buy_condition = (
        tendencia == "UP"
        and rsi >= 35
        and macd is not None and macd > -0.05
        and preco > ema9
        and high > prev_high * 1.0015  
    )


    sell_condition = (
        tendencia == "DOWN"
        and rsi <= 65
        and macd is not None and macd < 0.05
        and preco < ema9
        and low < prev_low * 0.9985   
    )


    if conf4h:
        if buy_condition:
            buy_condition = True
        if sell_condition:
            sell_condition = True


    if buy_condition:
        return montar_saida_agressiva(
            ticker, preco, "COMPRA",
            "Compra agressiva moderada: rompimento leve + EMA9 + RSI ok + MACD permissivo",
            rich
        )

    if sell_condition:
        return montar_saida_agressiva(
            ticker, preco, "VENDA",
            "Venda agressiva moderada: rompimento leve + EMA9 + RSI ok + MACD permissivo",
            rich
        )

    return montar_saida_agressiva(
        ticker, preco, "NEUTRO",
        "Sem confluências suficientes",
        rich
    )


def montar_saida_agressiva(ticker, preco, sinal, motivo, rich):
    return {
        "ticker": ticker,
        "preco": preco,
        "sinal": sinal,
        "motivo": motivo,
        "atr": rich.get("atr"),
        "rsi": rich.get("rsi"),
        "macd_hist": rich.get("macd_hist"),
        "volume_rel": rich.get("volume_rel"),
        "tendencia_base": rich.get("tendencia_base"),
        "confirm_4h": rich.get("confirm_4h"),
        "rr": rich.get("rr"),
        "rr_adjusted": rich.get("rr_adjusted"),
        "stops": rich.get("stops"),
    }


def check_4h_confirmation(ticker):


    df4 = carregar_dados(ticker, "4h")
    if df4 is None or len(df4) < 50:
        return False

    close = df4['Close'].iloc[-1]
    df4 = calcular_indicadores(df4)
    macd_hist = df4["MACD_HIST"].iloc[-1]
    ema9 = df4["Close"].ewm(span=9).mean().iloc[-1]

    # COMPRA
    if close > ema9 and macd_hist > 0:
        return True

    # VENDA
    if close < ema9 and macd_hist < 0:
        return True

    return False


def analisar_ticker(ticker, timeframe, usar_ml=True):

    intervalo = TIMEFRAMES.get(timeframe, "1d")
    df = carregar_dados(ticker, intervalo)
    if df is None or len(df) < 60:
        return {"ticker": ticker, "timeframe": timeframe, "sinal": None, "motivo": "Dados insuficientes"}

    if usar_ml:
        sinal_ml = obter_sinal_ml_do_banco(ticker)
        if sinal_ml is not None:
            return {
                "ticker": ticker,
                "timeframe": timeframe,
                "sinal": sinal_ml,
                "motivo": "Decisão baseada em ML"
            }


    df = calcular_indicadores(df)
    v = extrair_valores(df)

    
    sinal, motivo = regras_conservadoras_v2(df, v, ticker)

    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "sinal": sinal,
        "motivo": motivo,
        "preco": v["close"],
        "atr": v["atr"],
        "rsi": v["rsi"],
        "macd_hist": v["macd_hist"],
        "tendencia_base": "UP" if v["mm20"] > v["mm50"] else "DOWN" if v["mm20"] < v["mm50"] else "SIDE",
        "volume_rel": v["volume_rel"],
        "rr": None,
        "outcome": None
    }


def _calc_fast_indicators_local(df):
    return calcular_indicadores(df)



def regras_agressivas(ticker,
                      tf_main="1d",
                      tf_week="1w",
                      tf_confirm="4h",
                      vol_rel_threshold=0.5,
                      atr_multiplier=1.3,
                      rr_min=1.4,
                      allow_short=True):

    result = {
        "ticker": ticker,
        "tf_main": tf_main,
        "tf_week": tf_week,
        "tf_confirm": tf_confirm,
        "sinal": None,
        "motivo": None,
        "preco": None,
        "atr": None,
        "rsi": None,
        "macd_hist": None,
        "volume_rel": None,
        "weekly_trend": None,
        "confirm_4h": False,
        "stops": None,
        "rr": None,
        "rr_adjusted": None
    }


    df_week = carregar_dados(ticker, TIMEFRAMES.get(tf_week, tf_week))
    if df_week is None:
        result["motivo"] = "Sem dados semanais"
        return result

    df_week = calcular_indicadores(df_week)

    mm20_w = df_week["MM20"].iloc[-1]
    mm50_w = df_week["MM50"].iloc[-1]

    if np.isnan(mm20_w) or np.isnan(mm50_w):
        result["motivo"] = "Sem indicadores semanais suficientes"
        return result

    weekly_trend = "UP" if mm20_w > mm50_w else "DOWN" if mm20_w < mm50_w else "SIDE"
    result["weekly_trend"] = weekly_trend

    df_main = carregar_dados(ticker, TIMEFRAMES.get(tf_main, tf_main))
    if df_main is None or len(df_main) < 30:
        result["motivo"] = "Sem dados no timeframe principal"
        return result

    df_main = calcular_indicadores(df_main)

    close = float(df_main["Close"].iloc[-1])
    atr = float(df_main["ATR"].iloc[-1])
    rsi = float(df_main["RSI"].iloc[-1])
    macd_hist = float(df_main["MACD_HIST"].iloc[-1])
    vol_rel = float(df_main["VOL_REL"].iloc[-1])

    result["preco"] = close
    result["atr"] = atr
    result["rsi"] = rsi
    result["macd_hist"] = macd_hist
    result["volume_rel"] = vol_rel

    mm20 = df_main["MM20"].iloc[-1]
    mm50 = df_main["MM50"].iloc[-1]

    base = ticker.replace(".SA", "")
    BLUE_CHIPS = [s.replace(".SA","") for s in ACTIONS]


    if vol_rel is None:
        result["motivo"] = "Volume ausente"
        return result

    if vol_rel < vol_rel_threshold and base not in BLUE_CHIPS:
        result["motivo"] = f"Volume relativo baixo ({vol_rel:.2f}) — abaixo de {vol_rel_threshold}"
        return result

    tipo_padrao, nome_padrao = detectar_padrao_candle(df_main)

    buy_condition = False
    sell_condition = False


    if (
        (tipo_padrao == "COMPRA" or close > df_main["High"].rolling(10).max().shift(1).iloc[-1])
        and mm20 > mm50
        and (macd_hist > 0 or rsi > 50)
    ):
        buy_condition = True


    if (
        (tipo_padrao == "VENDA" or close < df_main["Low"].rolling(10).min().shift(1).iloc[-1])
        and mm20 < mm50
        and (macd_hist < 0 or rsi < 50)
    ):
        sell_condition = True



    if weekly_trend == "DOWN" and buy_condition:
        pass  

    if weekly_trend == "UP" and sell_condition and not allow_short:
        sell_condition = False


    confirm_ok = buy_condition or sell_condition

    df_4h = carregar_dados(ticker, TIMEFRAMES.get(tf_confirm, tf_confirm))
    if df_4h is not None and len(df_4h) > 20:
        df_4h = calcular_indicadores(df_4h)
        macd_4h = df_4h["MACD_HIST"].iloc[-1]
        close_4h = df_4h["Close"].iloc[-1]
        ema9_4h = df_4h["Close"].ewm(span=9).mean().iloc[-1]
        volrel_4h = df_4h["VOL_REL"].iloc[-1]

        if buy_condition and close_4h > ema9_4h and macd_4h > 0 and volrel_4h >= 0.6:
            confirm_ok = True

        if sell_condition and close_4h < ema9_4h and macd_4h < 0 and volrel_4h >= 0.6:
            confirm_ok = True

    result["confirm_4h"] = confirm_ok


    if (buy_condition or sell_condition) and confirm_ok:

        if atr is None or atr <= 0:
            result["motivo"] = "ATR inválido"
            return result

        if buy_condition:
            stop = max(df_main["Low"].tail(20).min(), close - atr * atr_multiplier)
            target = close + (close - stop) * 2
            trade_type = "COMPRA"
        else:
            stop = min(df_main["High"].tail(20).max(), close + atr * atr_multiplier)
            target = close - (stop - close) * 2
            trade_type = "VENDA"

        stop_pct = abs((close - stop) / close) * 100
        target_pct = abs((target - close) / close) * 100
        rr = target_pct / stop_pct if stop_pct > 0 else None
        rr_adj = rr * (atr / close) if rr else None

        result["sinal"] = trade_type
        result["rr"] = round(rr, 2) if rr else None
        result["rr_adjusted"] = round(rr_adj, 4) if rr_adj else None

        motivo = f"Agg setup: {trade_type}"
        motivo += f" (RR={rr:.2f})" if rr else " (RR indefinido)"
        if nome_padrao:
            motivo += f" | padrão={nome_padrao}"
        if weekly_trend == "DOWN" and trade_type == "COMPRA":
            motivo += " | Contra tendência semanal"
        if weekly_trend == "UP" and trade_type == "VENDA":
            motivo += " | Short contra semanal"

        result["motivo"] = motivo
        result["stops"] = {
            "stop": round(stop, 4),
            "target": round(target, 4),
            "stop_pct": round(stop_pct, 2),
            "target_pct": round(target_pct, 2)
        }

        return result


    result["motivo"] = "Sem setup agressivo confirmado"
    return result
