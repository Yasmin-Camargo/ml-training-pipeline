import os
import pandas as pd
from config.settings import DataConfig, ExperimentConfig
from .utils import log_message

def load_and_clean_data(filepath):
    """Loads data, removes nulls/duplicates/unwanted columns."""
    log_message(f"Loading data from {filepath}...", level="INFO")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep=DataConfig.CSV_SEPARATOR, low_memory=False)
    
    # Globally remove lines with FrameLevel 0 (Intra)
    if 'FrameLevel' in df.columns:
        initial_len = len(df)
        df = df[df['FrameLevel'] != 0]
        removed = initial_len - len(df)
        if removed > 0:
            log_message(f"Global Filter: Removed {removed} rows with FrameLevel == 0.", level="WARNING")

    # 0. Exclude specific rows based on config
    excluded_cfg = getattr(DataConfig, 'EXCLUDED_LINES', None)
    if excluded_cfg:
        col_name = excluded_cfg.get('collum_name')
        values = excluded_cfg.get('values', [])
        if values and col_name in df.columns:
            initial_len = len(df)
            df = df[~df[col_name].isin(values)]
            removed_count = initial_len - len(df)
            if removed_count > 0:
                log_message(f"Filtered out {removed_count} rows from excluded {col_name}: {values}", level="WARNING")

    # 1. Remove unwanted columns, but keep grouping column for grouped split later.
    if DataConfig.REMOVE_COLUMNS:
        group_col = getattr(DataConfig, 'GROUP_COLUMN', None)
        resolution_cols = getattr(DataConfig, 'RESOLUTION_COLUMNS', None)
        protected_cols = set()
        if group_col:
            protected_cols.add(group_col)
            if isinstance(resolution_cols, (list, tuple)):
                protected_cols.update(resolution_cols)

        cols_to_drop = [c for c in DataConfig.REMOVE_COLUMNS if c not in protected_cols]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        log_message(f"Removed columns: {cols_to_drop}", level="INFO")
        if protected_cols:
            log_message(
                f"Preserved protected columns for split: {sorted([c for c in protected_cols if c in df.columns])}",
                level="INFO"
            )

    # 2. Handle null values
    if DataConfig.TARGET_COLUMN in df.columns:
        if df[DataConfig.TARGET_COLUMN].isnull().any():
            before_target = len(df)
            df = df.dropna(subset=[DataConfig.TARGET_COLUMN])
            log_message(f"Dropped {before_target - len(df)} rows where Target '{DataConfig.TARGET_COLUMN}' was null.", level="WARNING")

    if df.isnull().values.any():
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        
        log_message("Detailed Null Report:", level="INFO")
        for col, count in cols_with_nulls.items():
            log_message(f"  -> Column '{col}': {count} missing values", level="DEBUG")

        if getattr(ExperimentConfig, 'IMPUTE_MISSING_VALUES', False):
            log_message("Config IMPUTE_MISSING_VALUES=True: Keeping rows with null features for later imputation.", level="INFO")
        else:
            log_message("Config IMPUTE_MISSING_VALUES=False: Removing rows with missing values.", level="WARNING")
            before = len(df)
            df = df.dropna()
            log_message(f"Removed {before - len(df)} rows.", level="WARNING")
        
    # 3. Handle Duplicates
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log_message(f"Removed {before - len(df)} duplicate rows.", level="WARNING")
    for col in df.select_dtypes(include='number').columns:
        clipped = df[col].clip(lower=-1e18, upper=1e18)
        if not clipped.equals(df[col]):
            log_message(f"Clipped values in column: {col}", level="DEBUG")
        df[col] = clipped
        
    # 5. Convert categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        mapping = {}   # dict<string, int>
        next_id = 0

        new_values = []
        for val in df[col].astype(str):
            if val not in mapping:
                mapping[val] = next_id
                next_id += 1
            new_values.append(mapping[val])

        df[col] = new_values
        log_message(f"Encoded '{col}' with mapping: {mapping}", level="DEBUG")
        
    return df
