# ===============================================
# LOGGER 
# ===============================================

import os
from datetime import datetime
import pytz
import logging

# São Paulo Timezone
FUSO = pytz.timezone("America/Sao_Paulo")

LOG_FILE = "log.txt"

def registrar_evento(mensagem, tipo="INFO"):
    
    try:
        agora = datetime.now(FUSO).strftime("%Y-%m-%d %H:%M:%S")
        linha = f"[{agora}] [{tipo}] {mensagem}\n"

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(linha)

        #
        print(linha.strip())

    except Exception as e:
        print(f"[ERRO LOGGER] Falha ao registrar log: {e}")




class LoggerHandler(logging.Handler):
    def emit(self, record):
        mensagem = record.getMessage()
        tipo = record.levelname
        registrar_evento(mensagem, tipo)

def limpar_log():
    
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        registrar_evento("Log reiniciado com sucesso.", "SYSTEM")
    except Exception as e:
        print(f"[ERRO LOGGER] Falha ao limpar log: {e}")
