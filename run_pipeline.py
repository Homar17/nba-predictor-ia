import subprocess
import sys
import os
import json
from datetime import datetime

# Importamos la funcion del archivo que acabamos de crear
from src.telegram_sender import send_telegram_message

def run_script(script_path):
    """Ejecuta un script de Python y detiene el proceso si hay un error."""
    print(f"\n{'-'*50}")
    print(f"Iniciando: {script_path}")
    print(f"{'-'*50}")
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Fallo en la ejecucion de {script_path}.")
        print("Deteniendo el pipeline para evitar propagar el error.")
        sys.exit(1)

def main():
    print("Iniciando el Pipeline Automático de Prediccion NBA IA...")
    
    # 1. Scripts base que SIEMPRE deben correr
    pipeline_scripts = [
        "src/data_collection.py",  
        "src/preprocessing.py",    
        "src/player_stats.py",     
        "src/injury_scraper.py",   
    ]
    
    # 2. Logica Inteligente de Entrenamiento
    acc_path = os.path.join("models", "accuracy.txt")
    needs_training = False
    dias_pasados = 0
    
    if not os.path.exists(acc_path):
        needs_training = True
    else:
        last_modified_time = os.path.getmtime(acc_path)
        last_modified_date = datetime.fromtimestamp(last_modified_time)
        dias_pasados = (datetime.now() - last_modified_date).days
        
        if dias_pasados >= 6:
            needs_training = True

    if needs_training:
        print("\n[INFO] Entrenamiento programado. Ejecutando Ajuste Fino.")
        pipeline_scripts.append("src/fine_tune.py")
    else:
        print(f"\n[INFO] Mantenimiento diario. (Ultimo entrenamiento hace {dias_pasados} dias).")
        
    for script in pipeline_scripts:
        run_script(script)
        
    # 3. Generar pronosticos y enviar a Telegram
    print(f"\n{'-'*50}")
    print("Calculando predicciones y formateando para Telegram...")
    print(f"{'-'*50}")
    
    try:
        # Ejecutamos predict y capturamos su salida de texto
        process = subprocess.run([sys.executable, "src/predict.py"], capture_output=True, text=True, check=True, encoding='utf-8')
        output = process.stdout
        
        if "---JSON_START---" in output and "---JSON_END---" in output:
            json_str = output.split("---JSON_START---")[1].split("---JSON_END---")[0].strip()
            games_data = json.loads(json_str)
            
            if not games_data:
                mensaje = "🏀 *NBA Predictor*\n\nNo hay partidos programados para hoy."
            else:
                mensaje = "🏀 *PRONOSTICOS NBA PARA HOY*\n\n"
                for game in games_data:
                    mensaje += f"*{game['away_team']} @ {game['home_team']}*\n"
                    mensaje += f"Gana: *{game['predicted_winner'].upper()}*\n"
                    
                    bajas_visita = ", ".join(game['away_injuries']) if game['away_injuries'] else "Ninguna"
                    bajas_local = ", ".join(game['home_injuries']) if game['home_injuries'] else "Ninguna"
                    
                    mensaje += f"Bajas {game['away_team']}: {bajas_visita}\n"
                    mensaje += f"Bajas {game['home_team']}: {bajas_local}\n"
                    mensaje += "—\n"
            
            # Enviar el texto armado por Telegram
            send_telegram_message(mensaje)
        else:
            print("[ERROR] No se pudo procesar el JSON de predict.py")
            
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Fallo en la ejecucion de predict.py.")
        sys.exit(1)

    print(f"\n{'-'*50}")
    print("PIPELINE AUTOMATICO COMPLETADO")
    print(f"{'-'*50}\n")

if __name__ == "__main__":
    main()