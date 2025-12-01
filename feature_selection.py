import time
from sklearn.feature_selection import RFECV
from logger import log_message
from config import RFECV_CV, RFECV_SCORING, RFECV_STEP, RFECV_MIN_FEATURES_SELECT, NUMBER_OF_JOBS
from model_config import BASE_MODEL_CONFIG

def recursive_feature_elimination_cv(X_train, y_train, classifier_name):
    start_time = time.time()
    log_message(f"Running RFECV with {classifier_name}...")
    
    if classifier_name not in BASE_MODEL_CONFIG:
         raise ValueError(f"Model {classifier_name} not configured in BASE_MODEL_CONFIG")

    model_conf = BASE_MODEL_CONFIG[classifier_name]
    estimator = model_conf['estimator'](**model_conf['params'])

    rfecv = RFECV(
        estimator=estimator,
        step=RFECV_STEP,
        cv=RFECV_CV,
        scoring=RFECV_SCORING,
        min_features_to_select=RFECV_MIN_FEATURES_SELECT,
        n_jobs=NUMBER_OF_JOBS
    )
    rfecv.fit(X_train, y_train)

    log_message(f"RFECV completed. Optimal number of features: {rfecv.n_features_}")
    selected_feature_names = X_train.columns[rfecv.support_].tolist()
    log_message(f"Selected features: {selected_feature_names}")
    
    feature_names = X_train.columns
    selected_features_with_ranking = [
        (str(name), int(rank))
        for name, rank, selected in zip(feature_names, rfecv.ranking_, rfecv.support_)
        if selected
    ]
    log_message(f"Selected features ranking: {selected_features_with_ranking}")
    
    
    log_message(f"Total time: {time.time() - start_time:.2f} seconds")

    return rfecv.support_
