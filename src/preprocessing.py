import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from config.settings import DataConfig, ExperimentConfig
from .utils import log_message


def _normalize_group_col(group_col):
    if group_col is None:
        return None
    if isinstance(group_col, str) and not group_col.strip():
        return None
    return group_col


def _random_group_split(df, target_col, group_col, test_size=0.25, random_state=42):
    """Split by unique groups without resolution stratification."""
    unique_groups = df[group_col].drop_duplicates().sort_values(kind='stable')
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
    )
    train_df = df[df[group_col].isin(train_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test


def split_by_video_resolution(
    df,
    target_col,
    test_size=0.25,
    random_state=42,
    group_col=None,
    width_col='FrameWidth',
    height_col='FrameHeight'
):
    """
    Split dataset keeping groups isolated and stratifying groups by resolution.
    """
    if group_col is None:
        group_col = getattr(DataConfig, 'GROUP_COLUMN', 'VideoName')
    group_col = _normalize_group_col(group_col)
    if group_col is None:
        raise ValueError("group_col cannot be empty for split_by_video_resolution.")

    required_cols = [group_col, width_col, height_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns required for stratified group split are missing: {missing_cols}"
        )

    # One row per group for leakage-safe split over groups only.
    video_info = df[[group_col, width_col, height_col]].drop_duplicates().copy()
    video_info['Resolution'] = (video_info[width_col] * video_info[height_col]).astype(str)

    # If a group appears with multiple resolutions, keep first to avoid duplicated
    # group ids in split candidates and log the condition.
    resolution_counts = video_info.groupby(group_col)['Resolution'].nunique()
    ambiguous_groups = resolution_counts[resolution_counts > 1]
    if not ambiguous_groups.empty:
        log_message(
            f"[SPLIT] Found {len(ambiguous_groups)} groups with multiple resolutions. "
            f"Using first observed resolution per group for stratification.",
            level="WARNING"
        )

    group_info = video_info.drop_duplicates(subset=[group_col], keep='first').copy()
    group_info = group_info.sort_values(by=[group_col], kind='stable')

    try:
        train_groups, test_groups = train_test_split(
            group_info[group_col],
            stratify=group_info['Resolution'],
            test_size=test_size,
            random_state=random_state
        )
    except ValueError as e:
        log_message(
            f"[SPLIT] Stratified split failed ({e}). Falling back to random group split.",
            level="WARNING"
        )
        train_groups, test_groups = train_test_split(
            group_info[group_col],
            test_size=test_size,
            random_state=random_state
        )

    train_df = df[df[group_col].isin(train_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()

    log_message(f"[SPLIT] Groups in Train ({group_col}): {len(train_groups)}", level="INFO")
    log_message(f"[SPLIT] Groups in Test ({group_col}): {len(test_groups)}", level="INFO")

    try:
        train_res = train_df.groupby([width_col, height_col])[group_col].nunique().to_dict()
        test_res = test_df.groupby([width_col, height_col])[group_col].nunique().to_dict()
        log_message(f"[SPLIT] Train resolution distribution (group counts): {train_res}", level="INFO")
        log_message(f"[SPLIT] Test resolution distribution (group counts): {test_res}", level="INFO")
    except Exception:
        # Non-blocking logging path.
        pass

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


def split_dataset_configurable(df, target_col, test_size=0.25, random_state=42, group_col=None):
    """
    Configurable dataset split.
    - Group mode (group_col valid): split by group, stratified by resolution.
    - Baseline mode (group_col empty/None): random row split.
    Returns groups_train for downstream CV (or None in baseline mode).
    """
    group_col = _normalize_group_col(group_col)

    if group_col and group_col in df.columns:
        log_message(f"[SPLIT] Group mode active using '{group_col}'.", level="INFO")
        resolution_cols = getattr(DataConfig, 'RESOLUTION_COLUMNS', None)
        use_resolution_stratification = isinstance(resolution_cols, (list, tuple)) and len(resolution_cols) >= 2

        if use_resolution_stratification:
            width_col = resolution_cols[0]
            height_col = resolution_cols[1]
            try:
                X_train, X_test, y_train, y_test = split_by_video_resolution(
                    df=df,
                    target_col=target_col,
                    test_size=test_size,
                    random_state=random_state,
                    group_col=group_col,
                    width_col=width_col,
                    height_col=height_col,
                )
            except ValueError as e:
                log_message(
                    f"[SPLIT] Resolution-stratified grouped split unavailable ({e}). "
                    f"Falling back to random split across groups.",
                    level="WARNING"
                )
                X_train, X_test, y_train, y_test = _random_group_split(
                    df=df,
                    target_col=target_col,
                    group_col=group_col,
                    test_size=test_size,
                    random_state=random_state,
                )
        else:
            log_message(
                "[SPLIT] RESOLUTION_COLUMNS not configured. Using random split across groups.",
                level="INFO"
            )
            X_train, X_test, y_train, y_test = _random_group_split(
                df=df,
                target_col=target_col,
                group_col=group_col,
                test_size=test_size,
                random_state=random_state,
            )

        groups_train = X_train[group_col].values if group_col in X_train.columns else None
    else:
        if group_col:
            log_message(
                f"[SPLIT] GROUP_COLUMN='{group_col}' not found. Falling back to baseline random split.",
                level="WARNING"
            )
        else:
            log_message("[SPLIT] Baseline mode active (GROUP_COLUMN empty).", level="WARNING")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
        )

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        groups_train = None

    return X_train, X_test, y_train, y_test, groups_train


def split_by_video_group(df, target_col, test_size=0.25, random_state=42, group_col=None):
    """Backward compatible alias for grouped split.

    Prefer split_dataset_configurable or split_by_video_resolution.
    """
    return split_by_video_resolution(
        df=df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        group_col=group_col,
    )


def sample_training_data(X_train, y_train, groups_train=None):
    """Create a stratified per-class sample from training data for tuning/RFECV."""
    train_data = pd.concat([X_train, y_train], axis=1)

    train_sampled = pd.concat([
        resample(
            g,
            replace=False,
            n_samples=min(len(g), ExperimentConfig.MAX_SAMPLES_PER_CLASS),
            random_state=ExperimentConfig.RANDOM_STATE
        )
        for _, g in train_data.groupby(DataConfig.TARGET_COLUMN)
    ])

    train_sampled = train_sampled.sort_index(kind='stable')
    X_train_samp = train_sampled.drop(columns=[DataConfig.TARGET_COLUMN])
    y_train_samp = train_sampled[DataConfig.TARGET_COLUMN]

    if groups_train is None:
        groups_train_samp = None
    else:
        if isinstance(groups_train, pd.Series):
            groups_series = groups_train
        else:
            groups_series = pd.Series(groups_train, index=X_train.index)
        groups_train_samp = groups_series.loc[X_train_samp.index].values

    return X_train_samp, y_train_samp, groups_train_samp

def balance_group_data(df_group):
    """Balances data within a specific block group."""
    target_col = DataConfig.TARGET_COLUMN
    
    if df_group[target_col].nunique() < 2:
        log_message(f"Group has less than 2 classes. Skipping balance.", level="WARNING")
        return df_group

    # Create composite key for balancing
    # Only use columns that exist in the dataframe
    valid_cols = [c for c in DataConfig.BALANCE_COLUMNS if c in df_group.columns]
    missing_cols = [c for c in DataConfig.BALANCE_COLUMNS if c not in df_group.columns]
    if missing_cols:
        log_message(f"Missing balance columns: {missing_cols}", level="WARNING")
    
    if not valid_cols:
        log_message(f"No valid columns found for balancing. Skipping balance.", level="ERROR")
        return df_group

    df_group['balance_key'] = list(zip(*(df_group[c] for c in valid_cols)))
    
    min_samples = df_group['balance_key'].value_counts().min()
    
    if min_samples < 1:
         return df_group.drop(columns=['balance_key'])

    balanced_df = pd.concat([
        resample(g, replace=False, n_samples=min_samples, random_state=ExperimentConfig.RANDOM_STATE)
        for _, g in df_group.groupby('balance_key')
    ])
    
    log_message(f"Group balanced. Total: {len(balanced_df)} samples.", level="INFO")
    return balanced_df.drop(columns=['balance_key'])


def normalize_data(X_train, X_test, X_train_samp):
    """Normalizes data using StandardScaler (fit on Train only)."""
    log_message("Normalizing data (fit on Train only)...", level="INFO")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    feature_names = X_train.columns.tolist()
    
    X_train_norm = pd.DataFrame(scaler.transform(X_train), columns=feature_names, index=X_train.index)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    X_train_samp_norm = pd.DataFrame(scaler.transform(X_train_samp), columns=feature_names, index=X_train_samp.index)
    
    means_str = ", ".join(f"{m:.6f}" for m in scaler.mean_)
    scales_str = ", ".join(f"{s:.6f}" for s in scaler.scale_) # scale_ é o desvio padrão
    
    log_message(f"Normalized params:", level="DEBUG")
    log_message(f"- Feature Order: {feature_names}", level="DEBUG")
    log_message(f"- means[] = {{ {means_str} }};", level="DEBUG")
    log_message(f"- scales[] = {{ {scales_str} }};", level="DEBUG")
    
    return X_train_norm, X_test_norm, X_train_samp_norm


def impute_data(X_train, X_test, X_train_samp):
    """Imputes missing values using a hybrid strategy: Mean for floats, Mode for ints/objects."""
    log_message("Imputing missing values ...", level="INFO")
    
    float_cols = X_train.select_dtypes(include=['float', 'float32', 'float64']).columns # Floats -> Average
    cat_cols = X_train.select_dtypes(include=['int', 'int32', 'int64', 'object', 'category']).columns  # Ints/Objects -> Mode
    
    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()
    X_train_samp_imp = X_train_samp.copy()
    
    impute_map = {}
    
    if len(float_cols) > 0:
        imputer_mean = SimpleImputer(strategy='mean')
        imputer_mean.fit(X_train[float_cols])
        
        X_train_imp[float_cols] = imputer_mean.transform(X_train[float_cols])
        X_test_imp[float_cols] = imputer_mean.transform(X_test[float_cols])
        X_train_samp_imp[float_cols] = imputer_mean.transform(X_train_samp[float_cols])
        
        for col, val in zip(float_cols, imputer_mean.statistics_):
            impute_map[col] = val

    if len(cat_cols) > 0:
        imputer_mode = SimpleImputer(strategy='most_frequent')
        imputer_mode.fit(X_train[cat_cols])
        
        X_train_imp[cat_cols] = imputer_mode.transform(X_train[cat_cols])
        X_test_imp[cat_cols] = imputer_mode.transform(X_test[cat_cols])
        X_train_samp_imp[cat_cols] = imputer_mode.transform(X_train_samp[cat_cols])
        
        for col, val in zip(cat_cols, imputer_mode.statistics_):
            impute_map[col] = val
            
    feature_names = X_train.columns.tolist()
    ordered_vals = [impute_map[col] for col in feature_names]
    
    vals_str = ", ".join(f"{v:.6f}" for v in ordered_vals)
    
    log_message("=== Imputation Parameters -  Mean (floats) / Mode (ints) ===", level="INFO")
    log_message(f"- Feature Order: {feature_names}", level="DEBUG")
    log_message(f"- impute_vals[] = {{ {vals_str} }};", level="DEBUG")
    
    if getattr(ExperimentConfig, 'IMPUTE_MISSING_VALUES', False):
        log_message(">> Applying imputation to datasets (Config=True).", level="WARNING")
        return X_train_imp, X_test_imp, X_train_samp_imp
    else:
        log_message(">> Skipping imputation application (Config=False). Using original data.", level="INFO")
        return X_train, X_test, X_train_samp


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