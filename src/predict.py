import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import json
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
        raise ValueError(f"No se encontraron datos para {team_abbr}")
    latest = team_games.iloc[-1]
    
    if latest['TEAM_ABBREVIATION_HOME'] == team_abbr:
        return {'REST_DAYS': latest['REST_DAYS_HOME'], 'ROAD_STREAK': 0, 'PTS': latest['PTS_LAST_5_HOME'], 'PLUS_MINUS': latest['PLUS_MINUS_LAST_5_HOME'], 'WIN': latest['WIN_LAST_5_HOME'], 'FG_PCT': latest['FG_PCT_LAST_5_HOME'], 'FG3_PCT': latest['FG3_PCT_LAST_5_HOME'], 'REB': latest['REB_LAST_5_HOME'], 'AST': latest['AST_LAST_5_HOME'], 'TOV': latest['TOV_LAST_5_HOME']}
    else:
        return {'REST_DAYS': latest['REST_DAYS_AWAY'], 'ROAD_STREAK': latest['ROAD_STREAK_AWAY'], 'PTS': latest['PTS_LAST_5_AWAY'], 'PLUS_MINUS': latest['PLUS_MINUS_LAST_5_AWAY'], 'WIN': latest['WIN_LAST_5_AWAY'], 'FG_PCT': latest['FG_PCT_LAST_5_AWAY'], 'FG3_PCT': latest['FG3_PCT_LAST_5_AWAY'], 'REB': latest['REB_LAST_5_AWAY'], 'AST': latest['AST_LAST_5_AWAY'], 'TOV': latest['TOV_LAST_5_AWAY']}

def get_h2h_win_pct(home_abbr, away_abbr, df_processed):
    matchups = df_processed[((df_processed['TEAM_ABBREVIATION_HOME'] == home_abbr) & (df_processed['TEAM_ABBREVIATION_AWAY'] == away_abbr)) | ((df_processed['TEAM_ABBREVIATION_HOME'] == away_abbr) & (df_processed['TEAM_ABBREVIATION_AWAY'] == home_abbr))]
    if matchups.empty: return 0.5
    latest_matchup = matchups.iloc[-1]
    return latest_matchup['H2H_WIN_PCT_HOME'] if latest_matchup['TEAM_ABBREVIATION_HOME'] == home_abbr else 1.0 - latest_matchup['H2H_WIN_PCT_HOME']

def calculate_injury_impact(team_name, df_averages):
    injury_file = os.path.join("data", "raw", "current_injuries.csv")
    if not os.path.exists(injury_file):
        return 0.0, 0, []
        
    df_injuries = pd.read_csv(injury_file)
    team_injuries = df_injuries[df_injuries['TEAM'].str.contains(team_name, case=False, na=False)]
    
    total_missing_ppg = 0.0
    players_out = len(team_injuries)
    injured_names = []
    
    for _, row in team_injuries.iterrows():
        injured_name_cbs = str(row['PLAYER'])
        injured_names.append(injured_name_cbs)
        match = df_averages[df_averages['PLAYER_NAME'].apply(lambda x: str(x) in injured_name_cbs)]
        
        if not match.empty:
            total_missing_ppg += match.iloc[0]['PPG']
        else:
            total_missing_ppg += 3.0
            
    return total_missing_ppg, players_out, injured_names

def predict_game(model, scaler, df_processed, df_averages, home_abbr, home_name, away_abbr, away_name):
    home_stats = get_team_rolling_stats(home_abbr, df_processed)
    away_stats = get_team_rolling_stats(away_abbr, df_processed)
    
    features_dict = {
        'H2H_WIN_PCT_HOME': get_h2h_win_pct(home_abbr, away_abbr, df_processed),
        'ROAD_STREAK_HOME': 0,
        'ROAD_STREAK_AWAY': away_stats['ROAD_STREAK'] + 1,
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
    
    feature_cols = ['H2H_WIN_PCT_HOME', 'ROAD_STREAK_HOME', 'ROAD_STREAK_AWAY', 'REST_ADVANTAGE', 'PTS_ADVANTAGE', 'PLUS_MINUS_ADVANTAGE', 'WIN_ADVANTAGE', 'FG_PCT_ADVANTAGE', 'FG3_PCT_ADVANTAGE', 'REB_ADVANTAGE', 'AST_ADVANTAGE', 'TOV_ADVANTAGE']
    
    features_scaled = scaler.transform(pd.DataFrame([features_dict], columns=feature_cols))
    base_prob_home = model.predict(features_scaled, verbose=0)[0][0]
    
    home_ppg, home_out, home_inj_list = calculate_injury_impact(home_name, df_averages)
    away_ppg, away_out, away_inj_list = calculate_injury_impact(away_name, df_averages)
    
    penalty_factor = 0.005
    adjusted_prob_home = max(0.01, min(0.99, base_prob_home - (home_ppg * penalty_factor) + (away_ppg * penalty_factor)))
    adjusted_prob_away = 1.0 - adjusted_prob_home
    
    return {
        "home_team": home_abbr,
        "home_name": home_name,
        "away_team": away_abbr,
        "away_name": away_name,
        "home_prob": round(adjusted_prob_home * 100, 2),
        "away_prob": round(adjusted_prob_away * 100, 2),
        "home_injuries": home_inj_list,
        "away_injuries": away_inj_list,
        "predicted_winner": home_name if adjusted_prob_home > 0.5 else away_name
    }

CBS_TEAM_NAMES = {"ATL": "Atlanta", "BOS": "Boston", "BKN": "Brooklyn", "CHA": "Charlotte", "CHI": "Chicago", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GSW": "Golden St.", "HOU": "Houston", "IND": "Indiana", "LAC": "L.A. Clippers", "LAL": "L.A. Lakers", "MEM": "Memphis", "MIA": "Miami", "MIL": "Milwaukee", "MIN": "Minnesota", "NOP": "New Orleans", "NYK": "New York", "OKC": "Oklahoma City", "ORL": "Orlando", "PHI": "Philadelphia", "PHX": "Phoenix", "POR": "Portland", "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto", "UTA": "Utah", "WAS": "Washington"}

if __name__ == "__main__":
    model_loaded, scaler_loaded = load_artifacts()
    df_proc = pd.read_csv(os.path.join("data", "processed", "nba_processed_games.csv"))
    df_avg = pd.read_csv(os.path.join("data", "raw", "player_averages.csv"))
    
    results = []
    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        if games:
            for game in games:
                h_abbr = game['homeTeam']['teamTricode']
                a_abbr = game['awayTeam']['teamTricode']
                results.append(predict_game(model_loaded, scaler_loaded, df_proc, df_avg, h_abbr, CBS_TEAM_NAMES.get(h_abbr, h_abbr), a_abbr, CBS_TEAM_NAMES.get(a_abbr, a_abbr)))
    except:
        pass
    
    # Imprime unicamente el formato JSON estructurado
    print("---JSON_START---")
    print(json.dumps(results))
    print("---JSON_END---")