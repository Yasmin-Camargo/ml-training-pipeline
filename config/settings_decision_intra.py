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
    FILE_PATH = DATA_DIR / "dataset_completo.csv"
    CSV_SEPARATOR = ';'
    TARGET_COLUMN = 'IntraKept'
    # 'VideoName' enables leakage-safe grouped pipeline; use '' or None for baseline random mode.
    GROUP_COLUMN = 'VideoName'
    RESOLUTION_COLUMNS = ['FrameWidth', 'FrameHeight']
    
    REMOVE_COLUMNS_CODEC = ['FinalDecision', 'IsIntra', 'IsSplit', 'VideoName', 'EncoderPreset', 'Frame', 'X_Pos', 'Y_Pos', 'FrameWidth', 'FrameHeight']
    
    REMOVE_COLUMNS = REMOVE_COLUMNS_CODEC
    
    EXCLUDED_LINES = {}
    
    # Columns used for balancing logic
    BALANCE_COLUMNS = ['IntraKept', 'FrameLevel']
    #BALANCE_COLUMNS = ['IsIntra', 'TargetQP', 'FrameWidth', 'FrameHeight']

# --- Experiment Configuration ---
class ExperimentConfig:
    RANDOM_STATE = 42
    N_JOBS = 5
    TEST_SIZE = 0.25
    MAX_SAMPLES_PER_CLASS = 20000
    NORMALIZE_DATA = True
    
    # Handling Missing Values
    # --> True: Impute missing values
    # --> False: Remove any row with missing values (drop)
    IMPUTE_MISSING_VALUES = False
    
    # Cross Validation
    CV_FOLDS = 5
    SCORING = 'accuracy'
    
    # Feature Selection (RFCV)
    RFE_ENABLED = True
    RFE_STEP = 1
    RFE_MIN_FEATURES = 5
    
    # Hyperparameter Tuning
    RANDOM_SEARCH_ITER = 2000
    
    # Flags
    RUN_VALIDATION_CURVES = False
    RUN_LEARNING_CURVES = False
    RUN_LEARNING_CURVES_AT_END = False
    LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]
    EXPORT_CPP = True
    
    # Active Grouping Strategies
    # Options: 'area', 'max', 'orientation', 'aspect_ratio', 'all', 'single'
    ACTIVE_GROUPINGS = ['single']
    
    #Options: 'frame_tier', 'neighborhood', 'block_size', 'qp', 'texture'
    #ACTIVE_GROUPINGS = ['single','frame_level', 'neighborhood', 'frame_tier', 'block_size', 'texture', 'qp']

