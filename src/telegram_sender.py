import requests

# Reemplaza con el Token que te dio BotFather
TOKEN = "8646431458:AAErjZ-5O_mr3EEnQBul8jAY7eWGAKaB-wk"

# Reemplaza con tu ID negativo obtenido de la API
CHAT_ID = "-1003825074646" 

def send_telegram_message(message):
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
        else:
            print(f"[ERROR TELEGRAM] Fallo en el envio: {response.text}")
    except Exception as e:
        print(f"[EXCEPCION TELEGRAM] No se pudo conectar: {e}")