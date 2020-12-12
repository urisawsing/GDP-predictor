import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(BASE_DIR, "db.sqlite3")

MODELS_PATH = os.path.join(BASE_DIR, "models")

LOGS_PATH = os.path.join(BASE_DIR, "logs")

NUM_PREDICTORS=50

GBM_HYP={
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'loss': 'ls',
       
}