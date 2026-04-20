import requests
import os
from dotenv import load_dotenv

# Carga las variables ocultas del archivo .env al sistema
load_dotenv()

# Extrae los valores de forma segura
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TOKEN or not CHAT_ID:
        print("[ERROR TELEGRAM] Faltan credenciales. Revisa tu archivo .env")
        return False
        
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("[TELEGRAM] Notificacion enviada correctamente a tu dispositivo.")
            return True
        else:
            print(f"[ERROR TELEGRAM] Fallo en el envio: {response.text}")
            return False
    except Exception as e:
        print(f"[EXCEPCION TELEGRAM] No se pudo conectar: {e}")
        return False