import subprocess
import sys

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
    print("Iniciando el Pipeline de Prediccion NBA IA con Aprendizaje Continuo...")
    
    # Lista de scripts en el orden exacto de ejecucion
    pipeline_scripts = [
        "src/data_collection.py",  # 1. Actualiza el historial con los resultados de ayer
        "src/preprocessing.py",    # 2. Calcula rachas, descansos y diferenciales netos
        "src/player_stats.py",     # 3. Descarga los promedios de puntos (PPG) actuales
        "src/injury_scraper.py",   # 4. Extrae los lesionados de hoy desde CBS
        "src/fine_tune.py",        # 5. El modelo aprende de los resultados de ayer
        "src/predict.py"           # 6. Cruza todo y emite los pronosticos de hoy
    ]
    
    for script in pipeline_scripts:
        run_script(script)
        
    print(f"\n{'-'*50}")
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"{'-'*50}\n")

if __name__ == "__main__":
    main()