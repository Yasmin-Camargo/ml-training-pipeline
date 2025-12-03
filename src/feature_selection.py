import time
from sklearn.feature_selection import RFECV
from config.settings import ExperimentConfig
from config.model_hyperparameters import BASE_MODELS
from .utils import log_message

def run_rfe(X, y, model_type):
    """Runs Recursive Feature Elimination."""
    if not ExperimentConfig.RFE_ENABLED:
        return X.columns.tolist()

    log_message(f"Running RFECV with {model_type}...")
    start_time = time.time()
    
    model_conf = BASE_MODELS[model_type]
    estimator = model_conf['estimator'](**model_conf['params'])
    
    rfecv = RFECV(
        estimator=estimator,
        step=ExperimentConfig.RFE_STEP,
        cv=ExperimentConfig.CV_FOLDS,
        scoring=ExperimentConfig.SCORING,
        min_features_to_select=ExperimentConfig.RFE_MIN_FEATURES,
        n_jobs=ExperimentConfig.N_JOBS
    )
    
    rfecv.fit(X, y)
    
    selected_features = X.columns[rfecv.support_].tolist()
    log_message(f"RFECV finished in {time.time() - start_time:.2f}s. Optimal number of features: {rfecv.n_features_}")
    log_message(f"Selected Features: {selected_features}")
    
    return selected_features