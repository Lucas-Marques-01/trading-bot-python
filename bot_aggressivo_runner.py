


import asyncio
import sqlite3
from datetime import datetime
import pytz
import discord
import logging
from ml_validator import ml_model_has_minimum_data
from bot_analise_tecnica import analisar_ticker_agressivo
from dotenv import load_dotenv
import os
load_dotenv()


from logger_bot import LoggerHandler
handler = LoggerHandler()
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)



from bot_analise_tecnica import (
    ACTIONS,
    SAO_PAULO_TZ,
)


# ===================== CONFIG ==============================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN_AGRESSIVO")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID_AGRESSIVO")
DB_FILE = os.getenv("DB_FILE")


logging.basicConfig(filename="log_aggressive.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ===================== DB Utils ==============================
def db_execute(query, params=(), fetch=True):
    conn = sqlite3.connect(DB_FILE, timeout=5, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    c = conn.cursor()
    c.execute(query, params)
    rows = c.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return rows


def get_last_alert_record(ticker):
    rows = db_execute("SELECT type, timestamp FROM alerts WHERE ticker=? ORDER BY id DESC LIMIT 1", (ticker,))
    return rows[0] if rows else (None, None)


def save_alert_aggressive(data):

    ts = datetime.now(SAO_PAULO_TZ).isoformat()
    db_execute(

        (
            data["ticker"],
            "agg",
            data.get("sinal") or "NEUTRO",
            data.get("preco"),
            data.get("motivo"),
            ts,
            data.get("atr"),
            data.get("rsi"),
            data.get("macd_hist"),
            data.get("tendencia_base"),
            data.get("volume_rel"),
            data.get("rr")
        ),
        fetch=False
    )


# ===================== EMBED DISCORD ========================
def build_aggressive_embed(data):
    ticker = data["ticker"].replace(".SA","")
    sinal = data.get("sinal")
    stops = data.get("stops") or {}


    # Cores
    if sinal == "COMPRA":
        title = f"🔵 COMPRA (AGRESSIVO) — {ticker}"
        color = 0x2ECC71
    elif sinal == "VENDA":
        title = f"🔴 SHORT (AGRESSIVO) — {ticker}"
        color = 0xE74C3C
    else:
        title = f"⚪ SEM ENTRADA — {ticker}"
        color = 0x95A5A6


    embed = discord.Embed(title=title, description=data.get("motivo", ""), color=color)


    # Basic
    embed.add_field(name="💰 Preço", value=f"R$ {data['preco']:.2f}" if data['preco'] else "--", inline=True)
    embed.add_field(name="📉 Tendência semanal", value=data.get("tendencia_base"), inline=True)
    embed.add_field(name="⏳ Confirmação 4H", value="✔️" if data.get("confirm_4h") else "❌", inline=True)


    
    embed.add_field(name="RSI7", value=f"{data['rsi']:.2f}" if data['rsi'] else "--", inline=True)
    embed.add_field(name="MACD Fast Hist", value=f"{data['macd_hist']:.4f}" if data['macd_hist'] else "--", inline=True)
    embed.add_field(name="Vol. Relativo", value=f"{data['volume_rel']:.2f}" if data['volume_rel'] else "--", inline=True)


    
    if sinal in ["COMPRA","VENDA"]:
        embed.add_field(name="🛑 Stop", value=f"R$ {stops.get('stop'):.2f}" if stops.get('stop') else "--", inline=True)
        embed.add_field(name="🎯 Target", value=f"R$ {stops.get('target'):.2f}" if stops.get('target') else "--", inline=True)
        embed.add_field(name="📊 RR", value=f"{data.get('rr')} (adj: {data.get('rr_adjusted')})", inline=False)


    return embed



async def run_aggressive_analysis():
    channel = client.get_channel(DISCORD_CHANNEL_ID)
    usar_ml = ml_model_has_minimum_data(min_total=200, min_each_class=30)
    
    for ticker in ACTIONS:
        try:
            data = analisar_ticker_agressivo(ticker, timeframe="1d", usar_ml=usar_ml)
            logging.info(f"Dados de análise agressiva para {ticker}: {data}")


            save_alert_aggressive(data)


            last = get_last_alert_record(ticker)
            last_type = last[0] if last else None
            if data.get("sinal") and data.get("sinal") != last_type:
                embed = build_aggressive_embed(data)
                await channel.send(embed=embed)
                logging.info(f"AGG enviado {ticker}: {data.get('sinal')}")
            else:
                logging.info(f"AGG não enviado {ticker} (sem mudança de sinal).")


            await asyncio.sleep(1)


        except Exception as e:
            logging.exception(f"Erro no agressivo {ticker}: {e}")



intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    logging.info(f"Aggressivo pronto: {client.user}")
    try:
        ch = client.get_channel(DISCORD_CHANNEL_ID)
        await ch.send("🔥 Bot Agressivo iniciado (com shorts).")
    except:
        pass
    client.loop.create_task(loop_aggressive())


async def loop_aggressive():
    while True:
        
        await run_aggressive_analysis()
        await asyncio.sleep(60 * 60)



async def run_aggressive_analysis_forced():
    channel = client.get_channel(DISCORD_CHANNEL_ID)
    usar_ml = ml_model_has_minimum_data(min_total=200, min_each_class=30)
    
    for ticker in ACTIONS:
        try:
            data = analisar_ticker_agressivo(ticker, timeframe="1d", usar_ml=usar_ml)
            logging.info(f"Dados de análise agressiva para {ticker}: {data}")
            save_alert_aggressive(data)
            embed = build_aggressive_embed(data)
            await channel.send(embed=embed)
            logging.info(f"AGG forçado enviado {ticker}: {data.get('sinal')}")
            await asyncio.sleep(1)
        except Exception as e:
            logging.exception(f"Erro na análise agressiva forçada de {ticker}: {e}")



@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.channel.id != DISCORD_CHANNEL_ID:
        return


    content = message.content.strip().lower()


    if content.startswith("!analise"):
        await message.channel.send("🧠 Executando análise agressiva manual (forçada) — aguarde...")
        await run_aggressive_analysis_forced()
        await message.channel.send("✅ Análise agressiva manual finalizada.")



if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
