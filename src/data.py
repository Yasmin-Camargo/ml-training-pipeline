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
        
    # 4. Clip numeric values
    for col in df.select_dtypes(include='number').columns:
        clipped = df[col].clip(lower=-1e18, upper=1e18)
        if not clipped.equals(df[col]):
            log_message(f"Clipped values in column: {col}")
        df[col] = clipped
        
    return df