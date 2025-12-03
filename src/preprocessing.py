import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from config.settings import DataConfig, ExperimentConfig
from .utils import log_message

def balance_group_data(df_group):
    """Balances data within a specific block group."""
    target_col = DataConfig.TARGET_COLUMN
    
    if df_group[target_col].nunique() < 2:
        log_message(f"! Group has less than 2 classes. Skipping balance.")
        return df_group

    # Create composite key for balancing
    # Only use columns that exist in the dataframe
    valid_cols = [c for c in DataConfig.BALANCE_COLUMNS if c in df_group.columns]
    missing_cols = [c for c in DataConfig.BALANCE_COLUMNS if c not in df_group.columns]
    if missing_cols:
        log_message(f"Missing balance columns: {missing_cols}")
    
    if not valid_cols:
        log_message(f"No valid columns found for balancing. Skipping balance.")
        return df_group

    df_group['balance_key'] = list(zip(*(df_group[c] for c in valid_cols)))
    
    min_samples = df_group['balance_key'].value_counts().min()
    
    if min_samples < 1:
         return df_group.drop(columns=['balance_key'])

    balanced_df = pd.concat([
        resample(g, replace=False, n_samples=min_samples, random_state=ExperimentConfig.RANDOM_STATE)
        for _, g in df_group.groupby('balance_key')
    ])
    
    log_message(f"Group balanced. Total: {len(balanced_df)} samples.")
    return balanced_df.drop(columns=['balance_key'])

def split_and_sample(df):
    """Splits into Train/Test and creates a smaller sample for Tuning."""
    X = df.drop(columns=[DataConfig.TARGET_COLUMN, 'BlockGroup'], errors='ignore')
    y = df[DataConfig.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=ExperimentConfig.TEST_SIZE, 
        random_state=ExperimentConfig.RANDOM_STATE, 
        stratify=y
    )
    
    # Sampling for faster Hyperparameter Search
    train_data = pd.concat([X_train, y_train], axis=1)
    train_sampled = pd.concat([
        resample(g, replace=False, 
                 n_samples=min(len(g), ExperimentConfig.MAX_SAMPLES_PER_CLASS),
                 random_state=ExperimentConfig.RANDOM_STATE)
        for _, g in train_data.groupby(DataConfig.TARGET_COLUMN)
    ])
    
    X_train_samp = train_sampled.drop(columns=[DataConfig.TARGET_COLUMN])
    y_train_samp = train_sampled[DataConfig.TARGET_COLUMN]
    
    return X_train, X_test, y_train, y_test, X_train_samp, y_train_samp