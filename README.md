# NBA Predictor IA

Pipeline automatizado de Machine Learning para predecir resultados de la NBA utilizando Redes Neuronales y Aprendizaje Continuo.

## Sobre el Proyecto
Este sistema recolecta datos históricos y en vivo de la NBA, procesa estadísticas avanzadas (diferenciales de eficiencia, fatiga de gira, días de descanso) y utiliza un modelo de Deep Learning (TensorFlow/Keras) para pronosticar los ganadores de los partidos del día. El modelo incluye un bucle de retroalimentación que le permite aprender de sus errores semanales (Fine-Tuning) y ajusta sus probabilidades basándose en reportes de lesiones en tiempo real.

## ⚙️ Requisitos
* Python 3.12
* Dependencias listadas en `requirements.txt`

## Instalación

1. Clona este repositorio:
   ```
   git clone <TU_URL_DE_GITHUB>
   cd nbaia
   ```
2. Crea y activa un entorno virtual:
    ```
    py -3.12 -m venv env_nba
    .\env_nba\Scripts\activate
    ```

3.  Instala las dependencias:
    ```
    pip install -r requirements.txt
    ```

## Estructura del Proyecto
* 'run_pipeline.py': Orquestador principal del sistema.

* 'src/data_collection.py': Descarga resultados historicos y de la noche anterior.

* 'src/preprocessing.py': Feature Engineering (diferenciales, rachas, Head-to-Head).

* 'src/player_stats.py': Actualiza los promedios de puntos (PPG) de los jugadores.

* 'src/injury_scraper.py': Extrae las lesiones del dia desde CBS Sports.

* 'src/train.py': Script para entrenar el modelo base desde cero (Uso esporadico).

* 'src/fine_tune.py': Ajuste fino del modelo con los resultados recientes (Aprendizaje continuo).

* 'src/predict.py': Consulta la cartelera en vivo de la NBA y emite las probabilidades finales.

* 'data/': Contiene los datasets crudos y procesados (Excluidos del control de versiones).

* 'models/': Almacena el modelo '.keras' y el escalador '.pkl' (Excluidos del control de versiones).