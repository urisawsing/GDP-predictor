"""Contains the directories of different files and the default number of predictors."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(BASE_DIR, "db.sqlite3")

MODELS_PATH = os.path.join(BASE_DIR, "models")

LOGS_PATH = os.path.join(BASE_DIR, "logs")

NUM_PREDICTORS = 50

EXHAUSTIVE_ITER=10
