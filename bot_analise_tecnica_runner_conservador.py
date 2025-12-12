# bot_analise_tecnica_runner_conservador.py


import asyncio
import sqlite3
import math
from datetime import datetime
import pytz
import discord
import logging
import os

from dotenv import load_dotenv

load_dotenv()


from ml_validator import ml_model_has_minimum_data


from logger_bot import LoggerHandler


handler = LoggerHandler()
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)



from bot_analise_tecnica import (
    carregar_dados,
    calcular_indicadores,
    detectar_padrao_candle,
    extrair_valores,
    analisar_ticker,
    ACTIONS,
    TIMEFRAMES,
    SAO_PAULO_TZ
)


from bot_analise_tecnica import regras_conservadoras_v2



DISCORD_TOKEN = os.getenv("DISCORD_TOKEN_CONSERVADOR")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID_CONSERVADOR")
DB_FILE = os.getenv("DB_FILE") 
print(f"Usando banco SQLite: {DB_FILE}")



INTERVALO_MINUTOS_ALERTA = 60


logging.basicConfig(filename="runner_conservador.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("runner_conservador")

print(f"[Runner] Usando DB: {os.path.abspath(DB_FILE)}")



def db_execute(query, params=(), fetch=True):
    conn = sqlite3.connect(DB_FILE, timeout=5, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    c = conn.cursor()
    c.execute(query, params)
    rows = c.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return rows


def get_last_alert_rich(ticker):
    """
    Retorna a última linha de alerts para o ticker com alguns campos úteis.
    Mantive o mesmo SELECT que você tinha.
    """
    row = db_execute(
        "SELECT type, atr, rsi, macd_hist, tendencia_base, volume_rel, rr, timestamp "
        "FROM alerts WHERE ticker=? ORDER BY id DESC LIMIT 1",
        (ticker,)
    )
    if not row:
        return None
    r = row[0]
    return {
        "type": r[0],
        "atr": r[1],
        "rsi": r[2],
        "macd_hist": r[3],
        "tendencia_base": r[4],
        "volume_rel": r[5],
        "rr": r[6],
        "timestamp": r[7]
    }


def save_alert_rich(data):
    """
    Salva registro OURO na tabela alerts.
    Mantive o esquema que você usava (campos OURO + outcome null).
    """
    ts = datetime.now(SAO_PAULO_TZ).isoformat()
    db_execute(
        """
        INSERT INTO alerts (ticker, timeframe, type, price, reason, timestamp, atr, rsi, macd_hist, tendencia_base, volume_rel, rr, outcome)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("ticker"),
            data.get("timeframe"),
            data.get("type"),
            data.get("price"),
            data.get("reason"),
            ts,
            data.get("atr"),
            data.get("rsi"),
            data.get("macd_hist"),
            data.get("tendencia_base"),
            data.get("volume_rel"),
            data.get("rr"),
            None
        ),
        fetch=False
    )
    logger.info(f"[DB] Alert saved (rich): {data.get('ticker')} {data.get('type')} price={data.get('price')} reason={data.get('reason')}")



def compute_strength_score(volume_rel, macd_hist, rr):

    vol = volume_rel if (volume_rel is not None and not math.isnan(volume_rel)) else 1.0
    mac = abs(macd_hist) if (macd_hist is not None and not math.isnan(macd_hist)) else 0.0
    rr_val = rr if (rr is not None and not math.isnan(rr)) else 0.0
    score = vol * (1.0 + mac) * (1.0 + (rr_val / 2.0))
    return score




def calc_stops_targets_from_df(df):

    df_local = df.copy()
    window = min(60, len(df_local))
    recent_high = float(df_local["High"].tail(window).max())
    recent_low = float(df_local["Low"].tail(window).min())
    price = float(df_local["Close"].iloc[-1])

    
    if "ATR" not in df_local.columns or df_local["ATR"].isnull().all():
        df2 = df_local.copy()
        df2["H-L"] = df2["High"] - df2["Low"]
        df2["H-PC"] = (df2["High"] - df2["Close"].shift()).abs()
        df2["L-PC"] = (df2["Low"] - df2["Close"].shift()).abs()
        df2["TR"] = df2[["H-L", "H-PC", "L-PC"]].max(axis=1)
        df2["ATR"] = df2["TR"].rolling(window=14).mean()
        atr_val = float(df2["ATR"].iloc[-1]) if not math.isnan(df2["ATR"].iloc[-1]) else None
    else:
        atr_val = float(df_local["ATR"].iloc[-1]) if not math.isnan(df_local["ATR"].iloc[-1]) else None

    if atr_val and atr_val > 0:
        atr_factor = 1.8  
        stop_buy = round(price - (atr_factor * atr_val), 2)
        target_buy = round(price + (atr_factor * atr_val * 2.2), 2)
        stop_sell = round(price + (atr_factor * atr_val), 2)
        target_sell = round(price - (atr_factor * atr_val * 2.2), 2)
    else:
        
        stop_buy = recent_low
        target_buy = recent_high
        stop_sell = recent_high
        target_sell = recent_low

    
    if price and stop_buy is not None:
        risk_buy = abs(price - stop_buy)
        stop_pct_buy = round((risk_buy / price) * 100, 2) if price else None
        target_pct_buy = round(((target_buy - price) / price) * 100, 2) if price else None
        rr_buy = round(target_pct_buy / stop_pct_buy, 2) if (stop_pct_buy and stop_pct_buy > 0) else None
    else:
        stop_pct_buy = target_pct_buy = rr_buy = None

    if price and stop_sell is not None:
        risk_sell = abs(stop_sell - price)
        stop_pct_sell = round((risk_sell / price) * 100, 2) if price else None
        target_pct_sell = round(((price - target_sell) / price) * 100, 2) if price else None
        rr_sell = round(target_pct_sell / stop_pct_sell, 2) if (stop_pct_sell and stop_pct_sell > 0) else None
    else:
        stop_pct_sell = target_pct_sell = rr_sell = None

    return {
        "stop_buy": stop_buy,
        "target_buy": target_buy,
        "stop_sell": stop_sell,
        "target_sell": target_sell,
        "support": recent_low,
        "resistance": recent_high,
        "atr": atr_val,
        "stop_pct_buy": stop_pct_buy,
        "target_pct_buy": target_pct_buy,
        "rr_buy": rr_buy,
        "stop_pct_sell": stop_pct_sell,
        "target_pct_sell": target_pct_sell,
        "rr_sell": rr_sell
    }



def multi_timeframe_confirmation_rich(ticker, usar_ml=True):


    details = {}
    result = {
        "ticker": ticker,
        "timeframe": "multi",
        "confirmed_signal": None,
        "reason_list": [],
        "details": {},
        "stops": None,
        
        "price": None,
        "atr": None,
        "rsi": None,
        "macd_hist": None,
        "tendencia_base": None,
        "volume_rel": None,
        "rr": None
    }


    for tf in ["1w", "1d"]:
        intervalo = TIMEFRAMES.get(tf)
        df = carregar_dados(ticker, intervalo)
        if df is None or df.empty:
            details[tf] = {"sinal": None, "motivo": "Dados insuficientes"}
            continue

        df = calcular_indicadores(df)  
        vals = extrair_valores(df)
        sinal, motivo = regras_conservadoras_v2(df, vals, ticker)

        details[tf] = {
            "sinal": sinal,
            "motivo": motivo,
            "preco": vals.get("close"),
            "atr": vals.get("atr"),
            "rsi": vals.get("rsi"),
            "macd_hist": vals.get("macd_hist"),
            "volume_rel": vals.get("volume_rel")
        }

    result["details"] = details


    tfd = details.get("1d") or {}
    tfw = details.get("1w") or {}


    reasons = []
    if tfd.get("sinal") == "COMPRA":
        if tfw.get("sinal") == "VENDA":
            reasons.append("Sem confirmação: semanal indica baixa (evitar operar contra macro).")
        else:
            result["confirmed_signal"] = "COMPRA"
            reasons.append("1d sinal de COMPRA e semanal não contradiz.")
    elif tfd.get("sinal") == "VENDA":
        if tfw.get("sinal") == "COMPRA":
            reasons.append("Sem confirmação: semanal indica alta (evitar operar contra macro).")
        else:
            result["confirmed_signal"] = "VENDA"
            reasons.append("1d sinal de VENDA e semanal não contradiz.")
    else:
        reasons.append("Sem alinhamento entre 1d e semanal — sem confirmação multi-timeframe.")

    result["reason_list"] = reasons


    df_1d = carregar_dados(ticker, TIMEFRAMES["1d"])
    if df_1d is None or df_1d.empty:
        return result

    df_1d = calcular_indicadores(df_1d)
    vals = extrair_valores(df_1d)

    result["price"] = vals.get("close")
    result["atr"] = vals.get("atr")
    result["rsi"] = vals.get("rsi")
    result["macd_hist"] = vals.get("macd_hist")
    
    result["tendencia_base"] = "UP" if df_1d["MM20"].iloc[-1] > df_1d["MM50"].iloc[-1] else ("DOWN" if df_1d["MM20"].iloc[-1] < df_1d["MM50"].iloc[-1] else "SIDE")
    result["volume_rel"] = vals.get("volume_rel")

    stops_obj = calc_stops_targets_from_df(df_1d)
    result["stops"] = stops_obj

    if result["confirmed_signal"] == "COMPRA":
        result["rr"] = stops_obj.get("rr_buy")
    elif result["confirmed_signal"] == "VENDA":
        result["rr"] = stops_obj.get("rr_sell")
    else:
        result["rr"] = None

    return result



def build_conservative_embed(rich):
    """
    Gera embed para Discord – versão aprimorada.
    Mantém seu layout original e adiciona seção premium para VENDA,
    destacando exatamente por que a venda foi confirmada.
    """

    ticker = rich["ticker"].replace(".SA", "")
    confirmed = rich.get("confirmed_signal")
    reasons = rich.get("reason_list") or []
    stops = rich.get("stops") or {}
    price = rich.get("price") or 0.0


    if confirmed == "COMPRA":
        title = f"🔵 CONSERVADOR — COMPRA CONFIRMADA — {ticker}"
        color = 0x2ECC71
    elif confirmed == "VENDA":
        title = f"🔴 CONSERVADOR — VENDA CONFIRMADA — {ticker}"
        color = 0xE74C3C
    else:
        title = f"⚪ CONSERVADOR — SEM SINAL — {ticker}"
        color = 0x95A5A6

    embed = discord.Embed(title=title, description="; ".join(reasons), color=color)


    desc = ""
    for tf in ["1w", "1d"]:
        d = rich.get("details", {}).get(tf, {})
        desc += f"**{tf}** — {d.get('sinal') or 'NEUTRO'} — {d.get('motivo') or ''}\n"
    embed.add_field(name="Resumo por timeframe", value=desc, inline=False)


    embed.add_field(name="Preço (1d close)", value=f"R$ {price:.2f}", inline=True)
    embed.add_field(name="Tendência semanal", value=rich.get("tendencia_base") or '--', inline=True)
    embed.add_field(name="Volume relativo (1d)", value=f"{rich.get('volume_rel') or '--'}", inline=True)


    embed.add_field(name="ATR", value=f"{rich.get('atr') or '--'}", inline=True)
    embed.add_field(name="RSI", value=f"{rich.get('rsi') or '--'}", inline=True)
    embed.add_field(name="MACD_HIST", value=f"{rich.get('macd_hist') or '--'}", inline=True)


    if confirmed == "VENDA":

        estrutura = rich.get("estrutura")
        breakout = rich.get("breakout") or {}
        pb = rich.get("pullback") or {}
        vol_rel = rich.get("volume_rel") or 0
        slope = rich.get("slope_mm20")
        macd = rich.get("macd_hist")
        conflu = rich.get("confluencia_venda") or 0
        vol_inc = rich.get("vol_inc", False)

        venda_msgs = []

        
        if estrutura == "BEAR":
            venda_msgs.append("✔️ Estrutura BEAR (LH/LL) confirmada")
        else:
            venda_msgs.append("❌ Sem estrutura BEAR — venda só passou por forte confluência")

        
        if pb.get("is_pullback"):
            venda_msgs.append(f"✔️ Pullback válido (FIB {pb.get('fib_hit')})")
        else:
            venda_msgs.append("❌ Sem pullback – evitou vender no fundo")

        
        if breakout.get("breakout_low"):
            venda_msgs.append("✔️ Breakout de mínima confirmado")
            
            if vol_rel > 1.5 and vol_inc:
                venda_msgs.append("🔥 Breakout baixo MUITO forte (volume_rel alto + ATR crescente)")
        else:
            venda_msgs.append("❌ Sem breakout de mínima")

        
        if macd is not None and macd < 0:
            venda_msgs.append("✔️ Momentum negativo (MACD < 0)")
        else:
            venda_msgs.append("❌ MACD não confirmou")

        
        if slope is not None and slope < 0:
            venda_msgs.append("✔️ Slope MM20 negativo (tendência curta de baixa)")
        else:
            venda_msgs.append("⚠️ Slope positivo — venda só passou por confluência extra / breakout forte")

        
        if vol_rel >= 0.6:
            venda_msgs.append(f"✔️ Volume vendedor adequado (VR={vol_rel:.2f})")
        else:
            venda_msgs.append("⚠️ Volume vendedor fraco — risco maior")

        
        venda_msgs.append(f"📊 Sinais de venda detectados: **{conflu}**")

        embed.add_field(
            name="📉 Por que a **VENDA** foi confirmada",
            value="\n".join(venda_msgs),
            inline=False
        )

    

    
    if stops:
        try:
            if confirmed == "COMPRA":
                embed.add_field(name="🛑 Stop (sugerido)", value=f"R$ {stops.get('stop_buy'):.2f}", inline=True)
                embed.add_field(name="🎯 Target (sugerido)", value=f"R$ {stops.get('target_buy'):.2f}", inline=True)
                embed.add_field(name="📌 Resistência (recent)", value=f"R$ {stops.get('resistance'):.2f}", inline=True)
            elif confirmed == "VENDA":
                embed.add_field(name="🛑 Stop (sugerido)", value=f"R$ {stops.get('stop_sell'):.2f}", inline=True)
                embed.add_field(name="🎯 Target (sugerido)", value=f"R$ {stops.get('target_sell'):.2f}", inline=True)
                embed.add_field(name="📌 Suporte (recent)", value=f"R$ {stops.get('support'):.2f}", inline=True)

            
            if confirmed == "COMPRA":
                embed.add_field(name="Stop (%)", value=f"-{stops.get('stop_pct_buy') or '--'}%", inline=True)
                embed.add_field(name="Target (%)", value=f"+{stops.get('target_pct_buy') or '--'}%", inline=True)
                embed.add_field(name="RR", value=f"{stops.get('rr_buy') or '--'}", inline=False)
            elif confirmed == "VENDA":
                embed.add_field(name="Stop (%)", value=f"-{stops.get('stop_pct_sell') or '--'}%", inline=True)
                embed.add_field(name="Target (%)", value=f"+{stops.get('target_pct_sell') or '--'}%", inline=True)
                embed.add_field(name="RR", value=f"{stops.get('rr_sell') or '--'}", inline=False)
        except Exception:
            logger.exception("[Embed] Erro ao montar stops/targets no embed")

    
    if not confirmed:
        embed.add_field(name="Por que sem sinal", value="; ".join(reasons), inline=False)

    return embed




async def run_full_analysis(force_send=False):
    """
    Mantive a lógica de:
     - calcular força do sinal
     - comparar com último alert salvo
     - re-enviar se mudou ou se força aumentou >=10%
     - salvar sempre o registro OURO na tabela alerts
    """
    channel = client.get_channel(DISCORD_CHANNEL_ID)
    usar_ml = ml_model_has_minimum_data(min_total=200, min_each_class=30)
    now = datetime.now(SAO_PAULO_TZ)
    logger.info(f"[Conservador] run_full_analysis (force={force_send}) em {now.isoformat()} usando ML: {usar_ml}")

    for ticker in ACTIONS:
        try:
            
            rich = multi_timeframe_confirmation_rich(ticker, usar_ml=usar_ml)

            confirmed = rich.get("confirmed_signal")
            reasons = rich.get("reason_list") or []
            price = rich.get("price")

            curr_strength = compute_strength_score(rich.get("volume_rel"), rich.get("macd_hist"), rich.get("rr"))

            last = get_last_alert_rich(ticker)
            last_type = last["type"] if last else None
            last_strength = compute_strength_score(last.get("volume_rel"), last.get("macd_hist"), last.get("rr")) if last else 0.0

            
            send_alert = False
            if confirmed and last_type != confirmed:
                send_alert = True  
            elif confirmed and last_type == confirmed and last_strength > 0 and curr_strength >= last_strength * 1.10:
                send_alert = True  
            elif force_send:
                send_alert = True  

            
            save_data = {
                "ticker": ticker,
                "timeframe": "multi",
                "type": confirmed if confirmed else "NEUTRO",
                "price": price,
                "reason": " | ".join(reasons) if reasons else "",
                "atr": rich.get("atr"),
                "rsi": rich.get("rsi"),
                "macd_hist": rich.get("macd_hist"),
                "tendencia_base": rich.get("tendencia_base"),
                "volume_rel": rich.get("volume_rel"),
                "rr": rich.get("rr")
            }
            save_alert_rich(save_data)

            
            if send_alert and channel and client.is_ready():
                embed = build_conservative_embed(rich)
                try:
                    await channel.send(embed=embed)
                    logger.info(f"[Conservador] Enviado {ticker}: {confirmed} (strength={curr_strength:.4f}, last={last_strength:.4f})")
                except Exception as e:
                    logger.exception(f"Erro ao enviar embed (conservador) {ticker}: {e}")
            else:
                logger.info(f"[Conservador] Não enviado {ticker} (send_alert={send_alert}) (confirmed={confirmed})")

            
            await asyncio.sleep(1.0)

        except Exception as e:
            
            logger.exception(f"Erro na análise conservadora de {ticker}: {e}")



intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def scheduled_loop():
    """
    Mantive a janela do pregão: seg-sex, 10:00-18:00 (America/Sao_Paulo)
    Isso evita sinais em after-hours (problema que você apontou).
    """
    await client.wait_until_ready()
    logger.info("Conservador scheduled_loop iniciado")
    while True:
        now = datetime.now(SAO_PAULO_TZ)
        
        if now.weekday() < 5 and 10 <= now.hour < 18:
            logger.info("Dentro do pregão — iniciando análise agendada (conservador).")
            await run_full_analysis(force_send=False)
        else:
            logger.info("Fora do pregão — conservador aguarda próxima janela.")
        await asyncio.sleep(INTERVALO_MINUTOS_ALERTA * 60)



@client.event
async def on_ready():
    logger.info(f"Conservador pronto — conectado como {client.user}")
    client.loop.create_task(scheduled_loop())
    try:
        ch = client.get_channel(DISCORD_CHANNEL_ID)
        if ch:
            await ch.send("🤖 Bot Conservador iniciado — operando apenas durante o pregão.")
    except Exception:
        logger.exception("Não foi possível enviar mensagem de inicialização (conservador).")


@client.event
async def on_message(message):

    if message.author == client.user:
        return
    if message.channel.id != DISCORD_CHANNEL_ID:
        return

    content = message.content.strip().lower()
    if content.startswith("!analise"):
        await message.channel.send("🧠 Executando análise manual (forçada) — aguarde...")
        await run_full_analysis(force_send=True)
        await message.channel.send("✅ Análise manual finalizada.")
    elif content.startswith("!status"):
        now = datetime.now(SAO_PAULO_TZ)
        await message.channel.send(f"🕒 Bot conservador ativo — horário SP: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    elif content.startswith("!ultimo"):
        parts = content.split()
        if len(parts) == 2:
            ticker = parts[1].upper()
            row = db_execute("SELECT timestamp, type, price, atr, rsi, macd_hist, volume_rel, rr FROM alerts WHERE ticker=? ORDER BY id DESC LIMIT 1", (ticker,))
            if row:
                ts, typ, price, atr, rsi, macd_hist, vol_rel, rr = row[0]
                await message.channel.send(f"Último sinal {ticker}: {typ} em {ts} — preço {price} | ATR={atr} RSI={rsi} MACD_H={macd_hist} VOL_REL={vol_rel} RR={rr}")
            else:
                await message.channel.send(f"Nenhum registro encontrado para {ticker}.")
        else:
            await message.channel.send("Use: !ultimo TICKER (ex: !ultimo PETR4)")



if __name__ == "__main__":
    logger.info("Iniciando runner conservador...")
    client.run(DISCORD_TOKEN)
