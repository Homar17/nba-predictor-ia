import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import regularizers

def load_data():
    input_file = os.path.join("data", "processed", "nba_processed_games.csv")
    print(f"Cargando datos procesados desde {input_file}...")
    df = pd.read_csv(input_file)
    return df

def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        
        # Mantenemos la arquitectura robusta pero ahora procesara caracteristicas mas limpias
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.4), 
        
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    df = load_data()
    
    # 1. Definir las nuevas columnas de entrada (12 caracteristicas de ingenieria)
    feature_cols = [
        'H2H_WIN_PCT_HOME',
        'ROAD_STREAK_HOME', 
        'ROAD_STREAK_AWAY',
        'REST_ADVANTAGE', 
        'PTS_ADVANTAGE', 
        'PLUS_MINUS_ADVANTAGE', 
        'WIN_ADVANTAGE',
        'FG_PCT_ADVANTAGE', 
        'FG3_PCT_ADVANTAGE', 
        'REB_ADVANTAGE', 
        'AST_ADVANTAGE', 
        'TOV_ADVANTAGE'
    ]
    
    # Verificacion de seguridad por si alguna columna falla en el preprocesamiento
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan estas columnas en el CSV: {missing_cols}")
        
    X = df[feature_cols]
    y = df['TARGET']
    
    # 2. Asignar pesos a las temporadas (Sample Weights) para priorizar el presente
    unique_seasons = sorted(df['SEASON_ID'].unique())
    print(f"Temporadas detectadas: {unique_seasons}")
    
    weight_step = 1.0 / len(unique_seasons)
    season_weights = {season: round((i + 1) * weight_step, 2) for i, season in enumerate(unique_seasons)}
    print(f"Pesos asignados por temporada: {season_weights}")
    
    w = df['SEASON_ID'].map(season_weights)
    
    # 3. Dividir los datos
    print("\nDividiendo y normalizando los datos...")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )
    
    # 4. Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entrenar
    print(f"Construyendo y entrenando la red neuronal con {X_train_scaled.shape[1]} caracteristicas netas...")
    model = build_model(X_train_scaled.shape[1])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        sample_weight=w_train,
        validation_data=(X_test_scaled, y_test, w_test),
        epochs=80,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 6. Evaluar
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n--- Resultados del Entrenamiento ---")
    print(f"Precision (Accuracy) final en datos de prueba: {accuracy * 100:.2f}%")
    
    # 7. Guardar artefactos
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model.save(os.path.join(models_dir, "nba_predictor.keras"))
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"Modelo y escalador guardados exitosamente en la carpeta '{models_dir}'.")

if __name__ == "__main__":
    main()