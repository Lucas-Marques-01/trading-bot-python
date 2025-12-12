import sqlite3
from dotenv import load_dotenv
import os

load_dotenv()

DB_FILE = os.getenv("DB_FILE")


def check_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tabelas encontradas no banco:")
    for table in tables:
        print("-", table[0])

    
    cursor.execute("PRAGMA table_info(alerts);")
    cols = cursor.fetchall()
    print("\nColunas da tabela 'alerts':")
    for col in cols:
        print(col)

    
    cursor.execute("SELECT DISTINCT ticker FROM alerts;")
    tickers = cursor.fetchall()
    if tickers:
        print("\nTickers encontrados na tabela 'alerts':")
        for t in tickers:
            print("-", t[0])
    else:
        print("\nNenhum ticker encontrado na tabela 'alerts'.")

    
    print("\nNúmero de linhas por ticker (up to 5 tickers):")
    for t in tickers[:5]:
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE ticker=?", (t[0],))
        count = cursor.fetchone()[0]
        print(f"{t[0]}: {count}")

    conn.close()

if __name__ == "__main__":
    check_db()
