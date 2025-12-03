import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
EXPORTS_DIR = BASE_DIR / "cpp_exports"
LOG_FILE = BASE_DIR / "execution.log"

# --- Data Configuration ---
class DataConfig:
    FILE_PATH = DATA_DIR / "features_10000_por_classe.csv"
    CSV_SEPARATOR = ','
    TARGET_COLUMN = 'MTSChosen'
    
    REMOVE_COLUMNS = ['VideoName']
    
    EXCLUDED_VIDEOS = []
    
    # Columns used for balancing logic
    BALANCE_COLUMNS = ['MTSChosen', 'cuQP', 'FrameWidth', 'FrameHeight']

# --- Experiment Configuration ---
class ExperimentConfig:
    RANDOM_STATE = 42
    N_JOBS = -1
    TEST_SIZE = 0.25
    MAX_SAMPLES_PER_CLASS = 100000
    NORMALIZE_DATA = False
    
    # Cross Validation
    CV_FOLDS = 5
    SCORING = 'accuracy'
    
    # Feature Selection (RFCV)
    RFE_ENABLED = True
    RFE_STEP = 1
    RFE_MIN_FEATURES = 5
    
    # Hyperparameter Tuning
    RANDOM_SEARCH_ITER = 100
    
    # Flags
    RUN_VALIDATION_CURVES = False
    RUN_LEARNING_CURVES = False
    LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]
    EXPORT_CPP = True
    
    # Active Grouping Strategies
    # Options: 'area', 'max', 'orientation', 'aspect_ratio', 'all', 'single'
    ACTIVE_GROUPINGS = ['single']
