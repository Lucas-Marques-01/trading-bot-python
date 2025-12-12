import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib
import os
import time


DB_FILE = "/home/ec2-user/bot-analise-tecnica-v0.4/alerts.db"
MODEL_DIR = "/home/ec2-user/bot-analise-tecnica-v0.4/models"


ACTIONS = ["PETR4.SA", "PETR3.SA", "VALE3.SA", "BBAS3.SA", "ITUB4.SA",
           "BBDC4.SA", "BPAC11.SA", "SUZB3.SA", "LREN3.SA", "WEGE3.SA", "ABEV3.SA"]


os.makedirs(MODEL_DIR, exist_ok=True)


def criar_tabela_ml_predictions():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        timestamp TEXT,
        predicted_type TEXT,
        confidence REAL
    )
    """)
    conn.commit()
    conn.close()


def load_data_for_ticker(ticker):
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(f"""
        SELECT * FROM alerts
        WHERE ticker = '{ticker}'
        ORDER BY timestamp ASC
    """, conn)
    conn.close()
    
    if df.empty:
        return df
    
    
    df['type'] = df['outcome'].astype(str).str.strip().str.upper()
    
    
    df = df[df['type'].isin(['COMPRA', 'VENDA'])]
    
    return df


def preprocess_data(df):
    df['type'] = df['type'].map({'COMPRA': 1, 'VENDA': 2})
    
    # Features derivadas
    df['return_1'] = df['price'].pct_change(1)
    df['return_5'] = df['price'].pct_change(5)
    df['atr_rsi_ratio'] = df['atr'] / (df['rsi'] + 1e-6)
    df['macd_rsi'] = df['macd_hist'] * df['rsi']
    
    df = df.dropna()
    
    feature_cols = ['price', 'atr', 'rsi', 'macd_hist', 'volume_rel', 'rr',
                    'return_1', 'return_5', 'atr_rsi_ratio', 'macd_rsi']
    X = df[feature_cols]
    y = df['type']
    
    return X, y


def train_and_evaluate(X, y, ticker, min_rows=50):
    if len(y) < min_rows:
        print(f"[{ticker}] Dados carregados: {len(y)} linhas. Insuficiente para treino ML.")
        return
    
    tscv = TimeSeriesSplit(n_splits=5)
    all_preds, all_true = [], []
    
    print(f"[{ticker}] Treinando com TimeSeriesSplit...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        all_preds.extend(preds)
        all_true.extend(y_test)
    
    print(f"[{ticker}] Relatório de classificação:")
    print(classification_report(all_true, all_preds))
    
    # Modelo final treinado em todos os dados
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    final_model.fit(X, y)
    
    model_path = os.path.join(MODEL_DIR, f'model_{ticker}.pkl')
    joblib.dump(final_model, model_path)
    print(f"[{ticker}] Modelo final salvo: {model_path}")


def save_ml_prediction(ticker, predicted_type, confidence):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO ml_predictions (ticker, timestamp, predicted_type, confidence)
        VALUES (?, datetime('now'), ?, ?)
    """, (ticker, predicted_type, confidence))
    conn.commit()
    conn.close()


def predict_signal(ticker, new_data):
    model_path = os.path.join(MODEL_DIR, f'model_{ticker}.pkl')
    if not os.path.exists(model_path):
        print(f"[{ticker}] Modelo não encontrado")
        return None
    
    model = joblib.load(model_path)
    
    
    new_data['return_1'] = new_data['price'].pct_change(1)
    new_data['return_5'] = new_data['price'].pct_change(5)
    new_data['atr_rsi_ratio'] = new_data['atr'] / (new_data['rsi'] + 1e-6)
    new_data['macd_rsi'] = new_data['macd_hist'] * new_data['rsi']
    new_data = new_data.fillna(method='ffill').fillna(0)
    
    feature_cols = ['price', 'atr', 'rsi', 'macd_hist', 'volume_rel', 'rr',
                    'return_1', 'return_5', 'atr_rsi_ratio', 'macd_rsi']
    X_new = new_data[feature_cols]
    
    probs = model.predict_proba(X_new)
    predicted_class_idx = probs[-1].argmax()
    predicted_type_num = model.classes_[predicted_class_idx]
    predicted_type = 'COMPRA' if predicted_type_num == 1 else 'VENDA'
    confidence = float(probs[-1][predicted_class_idx])
    
    save_ml_prediction(ticker, predicted_type, confidence)
    
    return probs[-1]


def main():
    criar_tabela_ml_predictions()
    while True:
        print("Iniciando ciclo de treino ML...")
        for ticker in ACTIONS:
            df = load_data_for_ticker(ticker)
            if df.empty:
                print(f"[{ticker}] Nenhum dado válido encontrado.")
                continue
            
            X, y = preprocess_data(df)
            train_and_evaluate(X, y, ticker)
        
        print("Ciclo de treino finalizado. Aguardando próximo ciclo...")
        
        time.sleep(6 * 60 * 60)

if __name__ == "__main__":
    main()
