import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime
from nba_api.live.nba.endpoints import scoreboard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_artifacts():
    model_path = os.path.join("models", "nba_predictor.keras")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def get_team_rolling_stats(team_abbr, df_processed):
    team_games = df_processed[(df_processed['TEAM_ABBREVIATION_HOME'] == team_abbr) | 
                              (df_processed['TEAM_ABBREVIATION_AWAY'] == team_abbr)].copy()
    
    if team_games.empty:
        raise ValueError(f"No se encontraron datos para el equipo {team_abbr}")
        
    latest = team_games.iloc[-1]
    
    if latest['TEAM_ABBREVIATION_HOME'] == team_abbr:
        stats = {
            'REST_DAYS': latest['REST_DAYS_HOME'],
            'ROAD_STREAK': 0,
            'PTS': latest['PTS_LAST_5_HOME'],
            'PLUS_MINUS': latest['PLUS_MINUS_LAST_5_HOME'],
            'WIN': latest['WIN_LAST_5_HOME'],
            'FG_PCT': latest['FG_PCT_LAST_5_HOME'],
            'FG3_PCT': latest['FG3_PCT_LAST_5_HOME'],
            'REB': latest['REB_LAST_5_HOME'],
            'AST': latest['AST_LAST_5_HOME'],
            'TOV': latest['TOV_LAST_5_HOME']
        }
    else:
        stats = {
            'REST_DAYS': latest['REST_DAYS_AWAY'],
            'ROAD_STREAK': latest['ROAD_STREAK_AWAY'],
            'PTS': latest['PTS_LAST_5_AWAY'],
            'PLUS_MINUS': latest['PLUS_MINUS_LAST_5_AWAY'],
            'WIN': latest['WIN_LAST_5_AWAY'],
            'FG_PCT': latest['FG_PCT_LAST_5_AWAY'],
            'FG3_PCT': latest['FG3_PCT_LAST_5_AWAY'],
            'REB': latest['REB_LAST_5_AWAY'],
            'AST': latest['AST_LAST_5_AWAY'],
            'TOV': latest['TOV_LAST_5_AWAY']
        }
    return stats

def get_h2h_win_pct(home_abbr, away_abbr, df_processed):
    matchups = df_processed[
        ((df_processed['TEAM_ABBREVIATION_HOME'] == home_abbr) & (df_processed['TEAM_ABBREVIATION_AWAY'] == away_abbr)) |
        ((df_processed['TEAM_ABBREVIATION_HOME'] == away_abbr) & (df_processed['TEAM_ABBREVIATION_AWAY'] == home_abbr))
    ]
    
    if matchups.empty:
        return 0.5
        
    latest_matchup = matchups.iloc[-1]
    
    if latest_matchup['TEAM_ABBREVIATION_HOME'] == home_abbr:
        return latest_matchup['H2H_WIN_PCT_HOME']
    else:
        return 1.0 - latest_matchup['H2H_WIN_PCT_HOME']

def calculate_injury_impact(team_name, df_averages):
    injury_file = os.path.join("data", "raw", "current_injuries.csv")
    if not os.path.exists(injury_file):
        return 0.0, 0
        
    df_injuries = pd.read_csv(injury_file)
    team_injuries = df_injuries[df_injuries['TEAM'].str.contains(team_name, case=False, na=False)]
    
    total_missing_ppg = 0.0
    players_out = len(team_injuries)
    
    for _, row in team_injuries.iterrows():
        injured_name_cbs = str(row['PLAYER'])
        match = df_averages[df_averages['PLAYER_NAME'].apply(lambda x: str(x) in injured_name_cbs)]
        
        if not match.empty:
            total_missing_ppg += match.iloc[0]['PPG']
        else:
            total_missing_ppg += 3.0
            
    return total_missing_ppg, players_out

def predict_game(model, scaler, df_processed, df_averages, home_team_abbr, home_team_name, away_team_abbr, away_team_name):
    home_stats = get_team_rolling_stats(home_team_abbr, df_processed)
    away_stats = get_team_rolling_stats(away_team_abbr, df_processed)
    
    road_streak_home = 0
    road_streak_away = away_stats['ROAD_STREAK'] + 1
    
    features_dict = {
        'H2H_WIN_PCT_HOME': get_h2h_win_pct(home_team_abbr, away_team_abbr, df_processed),
        'ROAD_STREAK_HOME': road_streak_home,
        'ROAD_STREAK_AWAY': road_streak_away,
        'REST_ADVANTAGE': home_stats['REST_DAYS'] - away_stats['REST_DAYS'],
        'PTS_ADVANTAGE': home_stats['PTS'] - away_stats['PTS'],
        'PLUS_MINUS_ADVANTAGE': home_stats['PLUS_MINUS'] - away_stats['PLUS_MINUS'],
        'WIN_ADVANTAGE': home_stats['WIN'] - away_stats['WIN'],
        'FG_PCT_ADVANTAGE': home_stats['FG_PCT'] - away_stats['FG_PCT'],
        'FG3_PCT_ADVANTAGE': home_stats['FG3_PCT'] - away_stats['FG3_PCT'],
        'REB_ADVANTAGE': home_stats['REB'] - away_stats['REB'],
        'AST_ADVANTAGE': home_stats['AST'] - away_stats['AST'],
        'TOV_ADVANTAGE': home_stats['TOV'] - away_stats['TOV']
    }
    
    feature_cols = [
        'H2H_WIN_PCT_HOME', 'ROAD_STREAK_HOME', 'ROAD_STREAK_AWAY', 'REST_ADVANTAGE', 
        'PTS_ADVANTAGE', 'PLUS_MINUS_ADVANTAGE', 'WIN_ADVANTAGE', 'FG_PCT_ADVANTAGE', 
        'FG3_PCT_ADVANTAGE', 'REB_ADVANTAGE', 'AST_ADVANTAGE', 'TOV_ADVANTAGE'
    ]
    
    features_df = pd.DataFrame([features_dict], columns=feature_cols)
    features_scaled = scaler.transform(features_df)
    
    base_prob_home = model.predict(features_scaled, verbose=0)[0][0]
    
    home_missing_ppg, home_out = calculate_injury_impact(home_team_name, df_averages)
    away_missing_ppg, away_out = calculate_injury_impact(away_team_name, df_averages)
    
    penalty_factor = 0.005
    adjusted_prob_home = base_prob_home - (home_missing_ppg * penalty_factor) + (away_missing_ppg * penalty_factor)
    adjusted_prob_home = max(0.01, min(0.99, adjusted_prob_home))
    adjusted_prob_away = 1.0 - adjusted_prob_home
    
    print(f"\n--- PREDICCION: {away_team_name} vs {home_team_name} ---")
    print(f"Impacto Lesiones Local     -> Jugadores fuera: {home_out} | PPG ausente: {home_missing_ppg:.1f}")
    print(f"Impacto Lesiones Visitante -> Jugadores fuera: {away_out} | PPG ausente: {away_missing_ppg:.1f}")
    print(f"Probabilidad de victoria (Local - {home_team_abbr}): {adjusted_prob_home * 100:.2f}%")
    print(f"Probabilidad de victoria (Visitante - {away_team_abbr}): {adjusted_prob_away * 100:.2f}%")
    
    if adjusted_prob_home > 0.5:
        print(f"PRONOSTICO: Gana {home_team_name}")
    else:
        print(f"PRONOSTICO: Gana {away_team_name}")

# Diccionario traductor: API -> Nombre en reporte de CBS
CBS_TEAM_NAMES = {
    "ATL": "Atlanta", "BOS": "Boston", "BKN": "Brooklyn", "CHA": "Charlotte",
    "CHI": "Chicago", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver",
    "DET": "Detroit", "GSW": "Golden St.", "HOU": "Houston", "IND": "Indiana",
    "LAC": "L.A. Clippers", "LAL": "L.A. Lakers", "MEM": "Memphis", "MIA": "Miami",
    "MIL": "Milwaukee", "MIN": "Minnesota", "NOP": "New Orleans", "NYK": "New York",
    "OKC": "Oklahoma City", "ORL": "Orlando", "PHI": "Philadelphia", "PHX": "Phoenix",
    "POR": "Portland", "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto",
    "UTA": "Utah", "WAS": "Washington"
}

if __name__ == "__main__":
    print("Cargando modelo y datos base...")
    model_loaded, scaler_loaded = load_artifacts()
    df_proc = pd.read_csv(os.path.join("data", "processed", "nba_processed_games.csv"))
    df_avg = pd.read_csv(os.path.join("data", "raw", "player_averages.csv"))
    
    print("\nConsultando la API oficial de la NBA para los juegos de hoy...")
    try:
        # Llamada al marcador en vivo (Corregido con la 'B' mayuscula)
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        
        if not games:
            print("No hay partidos programados para el dia de hoy.")
        else:
            for game in games:
                home_abbr = game['homeTeam']['teamTricode']
                away_abbr = game['awayTeam']['teamTricode']
                
                # Traducir los nombres para que coincidan con las lesiones de CBS
                home_name = CBS_TEAM_NAMES.get(home_abbr, home_abbr)
                away_name = CBS_TEAM_NAMES.get(away_abbr, away_abbr)
                
                predict_game(model_loaded, scaler_loaded, df_proc, df_avg, home_abbr, home_name, away_abbr, away_name)
                
    except Exception as e:
        print(f"Error al obtener los partidos del dia: {e}")