import pandas as pd
from sklearn.utils import resample
from logger import log_message
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TARGET_COLUMN, BALANCE_COLUMNS, TEST_SIZE

def balance_data(df, block_group):
    log_message(f"Balancing data for group '{block_group}'...")
    
    if 'BlockGroup' not in df.columns:
        raise ValueError("DataFrame must contain 'BlockGroup' column for grouping.")
    
    df_group = df[df['BlockGroup'] == block_group].copy()
    
    if df_group.empty:
        log_message(f"No data available for group '{block_group}'.")

    if df_group[TARGET_COLUMN].nunique() < 2:
        log_message(f"Only one class present for group '{block_group}' after filtering/labeling.")

    # 3. Lógica de balanceamento
    df_group['balance_key'] = list(zip(*(df_group[col] for col in BALANCE_COLUMNS)))

    min_samples_per_key = df_group['balance_key'].value_counts().min()
    if min_samples_per_key < 1:
        raise ValueError(f"Group '{block_group}' does not have sufficient combinations.")            
    balanced = pd.concat([
        resample(g, replace=False, n_samples=min_samples_per_key, random_state=RANDOM_STATE)
        for _, g in df_group.groupby('balance_key')
    ])

    log_message(f"Group '{block_group}' balanced. Total: {len(balanced)} samples.")

    return balanced.drop(columns=['BlockGroup', 'balance_key'])


def split_train_test(df):
    log_message("Splitting data into train and test sets...")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)