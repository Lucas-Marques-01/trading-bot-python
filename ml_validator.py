import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()

DB_FILE = os.getenv("DB_FILE") 


def ml_model_has_minimum_data(min_total=120, min_each_class=30):

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        
        cursor.execute("""
            SELECT COUNT(*) FROM alerts
            WHERE outcome IN ('GAIN', 'LOSS')
            AND type IN ('COMPRA', 'VENDA')
        """)
        total = cursor.fetchone()[0]

        
        cursor.execute("""
            SELECT outcome, COUNT(*)
            FROM alerts
            WHERE outcome IN ('GAIN', 'LOSS')
            AND type IN ('COMPRA', 'VENDA')
            GROUP BY outcome
        """)
        counts = dict(cursor.fetchall())

        conn.close()

        wins = counts.get('GAIN', 0)
        losses = counts.get('LOSS', 0)

        
        if total >= min_total and wins >= min_each_class and losses >= min_each_class:
            return True

        return False

    except Exception as e:
        print("[ERRO] No validador de ML:", e)
        return False
