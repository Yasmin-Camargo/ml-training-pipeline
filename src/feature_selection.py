import time
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold, KFold
from config.settings import ExperimentConfig
from config.model_hyperparameters import BASE_MODELS
from .utils import log_message

def run_rfe(X, y, model_type, groups=None):
    """Runs Recursive Feature Elimination."""
    if not ExperimentConfig.RFE_ENABLED:
        return X.columns.tolist()

    log_message(f"Running RFECV with {model_type}...", level="INFO")
    start_time = time.time()
    
    model_conf = BASE_MODELS[model_type]
    estimator = model_conf['estimator'](**model_conf['params'])
    
    if groups is not None:
        cv_strategy = GroupKFold(n_splits=ExperimentConfig.CV_FOLDS)
    else:
        cv_strategy = KFold(
            n_splits=ExperimentConfig.CV_FOLDS,
            shuffle=True,
            random_state=ExperimentConfig.RANDOM_STATE,
        )

    rfecv = RFECV(
        estimator=estimator,
        step=ExperimentConfig.RFE_STEP,
        cv=cv_strategy,
        scoring=ExperimentConfig.SCORING,
        min_features_to_select=ExperimentConfig.RFE_MIN_FEATURES,
        n_jobs=ExperimentConfig.N_JOBS
    )
    
    if groups is not None:
        rfecv.fit(X, y, groups=groups)
    else:
        rfecv.fit(X, y)
    
    selected_features = X.columns[rfecv.support_].tolist()
    
    final_estimator = rfecv.estimator_
    importances = None

    if hasattr(final_estimator, 'feature_importances_'):
        importances = final_estimator.feature_importances_
    elif hasattr(final_estimator, 'coef_'):
        importances = np.ravel(final_estimator.coef_)

    log_message(f"RFECV finished in {time.time() - start_time:.2f}s. Optimal number of features: {rfecv.n_features_}", level="INFO")
    log_message(f"Selected Features List: {selected_features}", level="DEBUG")

    if importances is not None:
        feature_score_pairs = list(zip(selected_features, importances))
        feature_score_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        log_message("=== Feature Importances (RFECV Final Model) ===", level="INFO")
        for feature_name, score in feature_score_pairs:
            log_message(f"Feature: {feature_name:<20} | Importance: {score:.6f}", level="INFO")

        max_features = getattr(ExperimentConfig, 'RFE_MAX_FEATURES', None)
        if max_features and len(feature_score_pairs) > max_features:
            log_message(f"Limiting selected features from {len(selected_features)} to the top {max_features}.", level="INFO")
            feature_score_pairs = feature_score_pairs[:max_features]
            selected_features = [feat for feat, score in feature_score_pairs]
    else:
        log_message("Could not extract feature importance/coefficients from this estimator.", level="WARNING")

        max_features = getattr(ExperimentConfig, 'RFE_MAX_FEATURES', None)
        if max_features and len(selected_features) > max_features:
            log_message(f"Limiting selected features from {len(selected_features)} to {max_features} (original RFECV order).", level="INFO")
            selected_features = selected_features[:max_features]
    
    return selected_features