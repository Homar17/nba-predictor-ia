import os
import pandas as pd
import numpy as np

def calculate_rolling_stats(df, window_size=5):
    """Calcula promedios, dias de descanso y fatiga de gira."""
    
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # 1. Calculo de dias de descanso
    df['REST_DAYS'] = df.groupby(['SEASON_ID', 'TEAM_ID'])['GAME_DATE'].diff().dt.days
    df['REST_DAYS'] = df['REST_DAYS'].fillna(3).clip(upper=7)
    
    # 2. Racha de partidos de visitante (Fatiga de gira)
    # Detectamos si es visitante buscando el '@' en el MATCHUP
    df['IS_AWAY'] = df['MATCHUP'].str.contains('@').astype(int)
    # Suma acumulativa de juegos de visita consecutivos (se reinicia cuando juegan en casa)
    df['ROAD_STREAK'] = df.groupby(['SEASON_ID', 'TEAM_ID', (df['IS_AWAY'] == 0).cumsum()])['IS_AWAY'].cumsum()
    
    # 3. Metricas de eficiencia
    metrics = ['PTS', 'PLUS_MINUS', 'WIN', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'TOV']
    for metric in metrics:
        col_name = f'{metric}_LAST_{window_size}'
        df[col_name] = df.groupby(['SEASON_ID', 'TEAM_ID'])[metric].transform(
            lambda x: x.shift(1).rolling(window=window_size).mean()
        )
        
    return df

def merge_home_away(df):
    """Une local y visitante y calcula los DIFERENCIALES NETOS."""
    
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
    
    df_home = df[df['IS_HOME']].copy()
    df_away = df[~df['IS_HOME']].copy()
    
    cols_to_keep = [
        'GAME_ID', 'SEASON_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 'WL',
        'REST_DAYS', 'ROAD_STREAK',
        'PTS_LAST_5', 'PLUS_MINUS_LAST_5', 'WIN_LAST_5',
        'FG_PCT_LAST_5', 'FG3_PCT_LAST_5', 'REB_LAST_5', 'AST_LAST_5', 'TOV_LAST_5'
    ]
    
    df_home = df_home[cols_to_keep]
    df_away = df_away[cols_to_keep]
    
    df_home['TARGET'] = df_home['WL'].apply(lambda x: 1 if x == 'W' else 0)
    df_home.drop(columns=['WL'], inplace=True)
    df_away.drop(columns=['WL'], inplace=True)
    
    df_game = pd.merge(df_home, df_away, on=['GAME_ID', 'SEASON_ID', 'GAME_DATE'], suffixes=('_HOME', '_AWAY'))
    
    # --- INGENIERIA DE CARACTERISTICAS (VENTAJAS NETAS) ---
    # Valores positivos indican ventaja para el LOCAL. Valores negativos, ventaja para la VISITA.
    df_game['REST_ADVANTAGE'] = df_game['REST_DAYS_HOME'] - df_game['REST_DAYS_AWAY']
    df_game['PTS_ADVANTAGE'] = df_game['PTS_LAST_5_HOME'] - df_game['PTS_LAST_5_AWAY']
    df_game['PLUS_MINUS_ADVANTAGE'] = df_game['PLUS_MINUS_LAST_5_HOME'] - df_game['PLUS_MINUS_LAST_5_AWAY']
    df_game['WIN_ADVANTAGE'] = df_game['WIN_LAST_5_HOME'] - df_game['WIN_LAST_5_AWAY']
    df_game['FG_PCT_ADVANTAGE'] = df_game['FG_PCT_LAST_5_HOME'] - df_game['FG_PCT_LAST_5_AWAY']
    df_game['FG3_PCT_ADVANTAGE'] = df_game['FG3_PCT_LAST_5_HOME'] - df_game['FG3_PCT_LAST_5_AWAY']
    df_game['REB_ADVANTAGE'] = df_game['REB_LAST_5_HOME'] - df_game['REB_LAST_5_AWAY']
    df_game['AST_ADVANTAGE'] = df_game['AST_LAST_5_HOME'] - df_game['AST_LAST_5_AWAY']
    df_game['TOV_ADVANTAGE'] = df_game['TOV_LAST_5_HOME'] - df_game['TOV_LAST_5_AWAY']
    
    return df_game

def add_h2h_stats(df_game, window=3):
    """Calcula el historial de enfrentamientos cara a cara."""
    
    df_game = df_game.sort_values('GAME_DATE')
    
    df_game['TEAM_A'] = df_game[['TEAM_ID_HOME', 'TEAM_ID_AWAY']].min(axis=1)
    df_game['TEAM_B'] = df_game[['TEAM_ID_HOME', 'TEAM_ID_AWAY']].max(axis=1)
    df_game['MATCHUP_PAIR'] = df_game['TEAM_A'].astype(str) + "_" + df_game['TEAM_B'].astype(str)
    
    df_game['TEAM_A_WON'] = ((df_game['TEAM_ID_HOME'] == df_game['TEAM_A']) & (df_game['TARGET'] == 1)) | \
                            ((df_game['TEAM_ID_AWAY'] == df_game['TEAM_A']) & (df_game['TARGET'] == 0))
    df_game['TEAM_A_WON'] = df_game['TEAM_A_WON'].astype(int)
    
    df_game['TEAM_A_WIN_PCT'] = df_game.groupby('MATCHUP_PAIR')['TEAM_A_WON'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    
    df_game['TEAM_A_WIN_PCT'] = df_game['TEAM_A_WIN_PCT'].fillna(0.5)
    
    df_game['H2H_WIN_PCT_HOME'] = np.where(
        df_game['TEAM_ID_HOME'] == df_game['TEAM_A'],
        df_game['TEAM_A_WIN_PCT'],
        1.0 - df_game['TEAM_A_WIN_PCT']
    )
    
    df_game = df_game.drop(columns=['TEAM_A', 'TEAM_B', 'MATCHUP_PAIR', 'TEAM_A_WON', 'TEAM_A_WIN_PCT'])
    
    return df_game

def main():
    input_file = os.path.join("data", "raw", "nba_raw_games.csv")
    output_dir = os.path.join("data", "processed")
    output_file = os.path.join(output_dir, "nba_processed_games.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Cargando datos crudos...")
    df = pd.read_csv(input_file)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    print("Calculando promedios y fatiga de gira...")
    df_with_stats = calculate_rolling_stats(df, window_size=5)
    
    print("Calculando diferenciales (Feature Engineering)...")
    df_final = merge_home_away(df_with_stats)
    
    print("Calculando historial cara a cara (Head-to-Head)...")
    df_final = add_h2h_stats(df_final, window=3)
    
    # Eliminamos nulos
    df_final = df_final.dropna()
    
    df_final.to_csv(output_file, index=False)
    print(f"Dataset procesado listo y guardado en: {output_file}")

if __name__ == "__main__":
    main()