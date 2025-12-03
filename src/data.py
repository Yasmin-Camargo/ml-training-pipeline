import os
import pandas as pd
from config.settings import DataConfig
from .utils import log_message

def load_and_clean_data(filepath):
    """Loads data, removes nulls/duplicates/unwanted columns."""
    log_message(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep=DataConfig.CSV_SEPARATOR, low_memory=False)

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
                log_message(f"Filtered out {removed_count} rows from excluded {col_name}: {values}")

    # 1. Remove unwanted columns
    if DataConfig.REMOVE_COLUMNS:
        df = df.drop(columns=DataConfig.REMOVE_COLUMNS, errors='ignore')
        log_message(f"Removed columns: {DataConfig.REMOVE_COLUMNS}")

    # 2. Handle Nulls
    if df.isnull().values.any():
        before = len(df)
        df = df.dropna()
        log_message(f"Removed {before - len(df)} rows with null values.")

    # 3. Handle Duplicates
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log_message(f"Removed {before - len(df)} duplicate rows.")
    for col in df.select_dtypes(include='number').columns:
        clipped = df[col].clip(lower=-1e18, upper=1e18)
        if not clipped.equals(df[col]):
            log_message(f"Clipped values in column: {col}")
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
        log_message(f"Encoded '{col}' with mapping: {mapping}")
        
    return df