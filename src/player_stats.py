import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

def fetch_player_averages():
    print("Descargando promedios por partido reales...")
    try:
        # Se agrega per_mode_detailed='PerGame' para evitar puntos totales
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
        
        # Filtramos las columnas necesarias
        df_clean = df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS']].copy()
        df_clean.rename(columns={'PTS': 'PPG'}, inplace=True)
        
        return df_clean
    except Exception as e:
        print(f"Error al descargar estadisticas de jugadores: {e}")
        return None

def main():
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "player_averages.csv")
    
    df_players = fetch_player_averages()
    
    if df_players is not None:
        df_players.to_csv(output_file, index=False)
        print(f"Promedios guardados exitosamente en: {output_file}")

if __name__ == "__main__":
    main()