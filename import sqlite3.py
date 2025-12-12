import sqlite3
import pandas as pd

DB_FILE = "/home/ec2-user/bot-analise-tecnica-v0.4/alerts.db"

ACTIONS = ["PETR4.SA", "PETR3.SA", "VALE3.SA", "BBAS3.SA", "ITUB4.SA",
           "BBDC4.SA", "BPAC11.SA", "SUZB3.SA", "LREN3.SA", "WEGE3.SA", "ABEV3.SA"]

conn = sqlite3.connect(DB_FILE)

for ticker in ACTIONS:
    df = pd.read_sql_query(f"SELECT ticker, type, outcome FROM alerts WHERE ticker='{ticker}' LIMIT 5", conn)
    print(f"\n[{ticker}] primeiros registros:")
    print(df)

conn.close()
