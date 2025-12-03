from config.settings import DataConfig

def _process_model_A(df):
    """
    Model A:
    - No filtering.
    - Labels: (0 or 1) -> 10, others -> 20.
    """
    df_out = df.copy()
    df_out[DataConfig.TARGET_COLUMN] = df_out[DataConfig.TARGET_COLUMN].apply(
        lambda x: 10 if x in [0, 1] else 20
    )
    return df_out

def _process_model_B(df):
    """
    Model B:
    - Remove rows where Target is 0 or 1.
    - Labels: (2) -> 10, others -> 20.
    """
    df_out = df[~df[DataConfig.TARGET_COLUMN].isin([0, 1])].copy()
    df_out[DataConfig.TARGET_COLUMN] = df_out[DataConfig.TARGET_COLUMN].apply(
        lambda x: 10 if x == 2 else 20
    )
    return df_out

def _process_logistic_raw(df):
    """
    - Dados brutos (sem filtro, sem renomear labels).
    """
    return df.copy()

# Scenario Definitions including dynamic classifier type
MODEL_STRATEGIES = {
    "decision_tree": {
        "description": "Model C (Logistic Regression - Raw Data)",
        "process_function": _process_logistic_raw,
        "classifier_type": "decision_tree"
    }
}

"""
MODEL_STRATEGIES = {
    "logistic_regression": {
        "description": "Model C (Logistic Regression - Raw Data)",
        "process_function": _process_logistic_raw,
        "classifier_type": "logistic_regression"
    }
}
"""

"""
MODEL_STRATEGIES = {
    "A": {
        "description": "Model A (Filter after DCT2)",
        "process_function": _process_model_A,
        "classifier_type": "decision_tree"
    },
    "B": {
        "description": "Model B (DST7 vs Others)",
        "process_function": _process_model_B,
        "classifier_type": "decision_tree"
    }
}
"""