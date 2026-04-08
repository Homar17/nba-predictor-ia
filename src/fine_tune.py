import os
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("Iniciando proceso de Ajuste Fino (Fine-Tuning)...")
    
    # 1. Cargar datos actualizados
    data_file = os.path.join("data", "processed", "nba_processed_games.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError("No se encontro el dataset procesado.")
    
    df = pd.read_csv(data_file)
    
    feature_cols = [
        'H2H_WIN_PCT_HOME', 'ROAD_STREAK_HOME', 'ROAD_STREAK_AWAY', 'REST_ADVANTAGE', 
        'PTS_ADVANTAGE', 'PLUS_MINUS_ADVANTAGE', 'WIN_ADVANTAGE', 'FG_PCT_ADVANTAGE', 
        'FG3_PCT_ADVANTAGE', 'REB_ADVANTAGE', 'AST_ADVANTAGE', 'TOV_ADVANTAGE'
    ]
    
    X = df[feature_cols]
    y = df['TARGET']
    
    # Pesos por temporada (dando prioridad a la temporada actual)
    unique_seasons = sorted(df['SEASON_ID'].unique())
    weight_step = 1.0 / len(unique_seasons)
    season_weights = {season: round((i + 1) * weight_step, 2) for i, season in enumerate(unique_seasons)}
    w = df['SEASON_ID'].map(season_weights)
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.1, random_state=42 # Usamos el 90% de los datos para actualizar
    )
    
    # 2. Cargar artefactos existentes
    model_path = os.path.join("models", "nba_predictor.keras")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("No se encontraron los modelos base. Ejecuta train.py primero.")
        
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    # Re-escalar datos con el escalador existente
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Compilar el modelo con una Tasa de Aprendizaje muy baja
    # Esto evita que el modelo de saltos bruscos y olvide el entrenamiento original
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Entrenando el modelo con los ultimos resultados de la NBA...")
    
    # Entrenamos por solo 10 epocas
    model.fit(
        X_train_scaled, y_train,
        sample_weight=w_train,
        validation_data=(X_test_scaled, y_test, w_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # 4. Evaluar y guardar
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n--- Resultados del Ajuste Fino ---")
    print(f"Nueva precision del modelo: {accuracy * 100:.2f}%")
    
    model.save(model_path)
    print("Modelo actualizado guardado exitosamente.")

if __name__ == "__main__":
    main()