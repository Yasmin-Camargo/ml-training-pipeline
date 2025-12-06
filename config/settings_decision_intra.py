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
    FILE_PATH = DATA_DIR / "features_from_image.csv"
    CSV_SEPARATOR = ';'
    TARGET_COLUMN = 'IsIntra'
    
    REMOVE_COLUMNS_CODEC = ['VideoName', 'EncoderPreset', 'Frame', 'X_Pos', 'Y_Pos']
    REMOVE_COLUMNS_IMAGE = [
        'blk_pixel_mean',
        'blk_pixel_variance',
        'blk_pixel_std_dev',
        'blk_pixel_sum',
        'blk_var_h',
        'blk_var_v',
        'blk_std_v',
        'blk_std_h',
        'blk_min',
        'blk_max',
        'blk_range',
        'blk_laplacian_var',
        'blk_entropy',
        'blk_sobel_gv',
        'blk_sobel_gh',
        'blk_sobel_mag',
        'blk_sobel_dir',
        'blk_sobel_razao_grad',
        'blk_prewitt_gv',
        'blk_prewitt_gh',
        'blk_prewitt_mag',
        'blk_prewitt_dir',
        'blk_prewitt_razao_grad',
        'blk_had_dc',
        'blk_had_energy_total',
        'blk_had_energy_ac',
        'blk_had_max',
        'blk_had_min',
        'blk_had_topleft',
        'blk_had_topright',
        'blk_had_bottomleft',
        'blk_had_bottomright',
    ]
    REMOVE_COLUMNS = REMOVE_COLUMNS_CODEC
    
    EXCLUDED_LINES = {
        "collum_name": "VideoName",
        "values": [
            "ParkRunning3", 
            "BasketballDrive", 
            "BasketballDrill", 
            "BasketballPass", 
            "KristenAndSara"
        ]
    }
    
    # Columns used for balancing logic
    BALANCE_COLUMNS = ['IsIntra', 'TargetQP', 'FrameWidth', 'FrameHeight']

# --- Experiment Configuration ---
class ExperimentConfig:
    RANDOM_STATE = 42
    N_JOBS = -1
    TEST_SIZE = 0.25
    MAX_SAMPLES_PER_CLASS = 200000
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
    RUN_LEARNING_CURVES_AT_END = True
    LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]
    EXPORT_CPP = True
    
    # Active Grouping Strategies
    # Options: 'area', 'max', 'orientation', 'aspect_ratio', 'all', 'single'
    ACTIVE_GROUPINGS = ['single']
