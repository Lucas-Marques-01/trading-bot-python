import sqlite3

DB_FILE = "alerts.db"

def cria_tabela_alerts():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Cria a tabela alerts com todas as colunas do padrão OURO, se não existir
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY,
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
        outcome TEXT
    );
    """)

    conn.commit()
    conn.close()
    print("✔ Tabela alerts criada ou já existente com esquema padrão OURO.")

def atualiza_colunas_alerts():
    required_columns = {
        "atr": "REAL",
        "rsi": "REAL",
        "macd_hist": "REAL",
        "tendencia_base": "TEXT",
        "volume_rel": "REAL",
        "rr": "REAL",
        "outcome": "TEXT"
    }

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(alerts)")
    existing_columns = [row[1] for row in cursor.fetchall()]

    for col, col_type in required_columns.items():
        if col not in existing_columns:
            print(f"Adicionando coluna: {col}...")
            cursor.execute(f"ALTER TABLE alerts ADD COLUMN {col} {col_type};")
        else:
            print(f"Coluna {col} já existe.")

    conn.commit()
    conn.close()
    print("✔ Atualização das colunas concluída.")

if __name__ == "__main__":
    cria_tabela_alerts()
    atualiza_colunas_alerts()
