import os
import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamelog

def fetch_seasons_data(seasons):
    all_games = []
    
    for season in seasons:
        print(f"Descargando datos de la temporada {season}...")
        try:
            # El endpoint trae SEASON_ID por defecto
            gamelog = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star='Regular Season'
            )
            df = gamelog.get_data_frames()[0]
            all_games.append(df)
            
            # Pausa de 2 segundos vital para evitar que la API de la NBA bloquee nuestra IP
            time.sleep(2)
            
        except Exception as e:
            print(f"Error descargando la temporada {season}: {e}")
            
    if all_games:
        # Unimos todas las temporadas en un solo DataFrame
        return pd.concat(all_games, ignore_index=True)
    return pd.DataFrame()

def main():
    # Lista de las ultimas 5 temporadas para dar volumen al modelo
    seasons_to_fetch = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "nba_raw_games.csv")
    
    print("Iniciando recoleccion masiva de datos historicos...")
    df_all_seasons = fetch_seasons_data(seasons_to_fetch)
    
    if not df_all_seasons.empty:
        # Asegurarnos de mantener SEASON_ID en la lista de columnas
        cols_to_keep = [
            'SEASON_ID', 'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 
            'MATCHUP', 'WL', 'PTS', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'TOV'
        ]
        
        # Filtramos asegurando que las columnas existen en la respuesta de la API
        available_cols = [col for col in cols_to_keep if col in df_all_seasons.columns]
        df_clean = df_all_seasons[available_cols].copy()
        
        df_clean.to_csv(output_file, index=False)
        print(f"\nDatos guardados exitosamente en: {output_file}")
        print(f"Total de partidos historicos listos para procesar: {len(df_clean)}")
    else:
        print("\nNo se pudieron descargar los datos.")

if __name__ == "__main__":
    main()