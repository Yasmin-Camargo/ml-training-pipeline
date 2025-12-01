import time
from sklearn.model_selection import RandomizedSearchCV
from logger import log_message
from config import RANDOM_STATE, CROSS_VALIDATION, SCORING, RANDOM_SEARCH_ITERATIONS, NUMBER_OF_JOBS
from model_config import MODEL_HYPERPARAMS, BASE_MODEL_CONFIG

def random_search(X_train, y_train, classifier_name):
    start_time = time.time()
    log_message(f"Starting randomized hyperparameter search for {classifier_name}...")

    if classifier_name not in BASE_MODEL_CONFIG or classifier_name not in MODEL_HYPERPARAMS:
         raise ValueError(f"Model {classifier_name} not configured in BASE_MODEL_CONFIG or MODEL_HYPERPARAMS")
         
    model_conf = BASE_MODEL_CONFIG[classifier_name]
    estimator_class = model_conf['estimator']
    base_params = model_conf['params']
    
    estimator = estimator_class(**base_params)

    param_dist = MODEL_HYPERPARAMS.get(classifier_name)

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        cv=CROSS_VALIDATION,
        random_state=RANDOM_STATE,
        scoring=SCORING,
        n_jobs=NUMBER_OF_JOBS
    )

    random_search.fit(X_train, y_train)
    log_message(f"Best parameters found: {random_search.best_params_}")
    log_message(f"Best cross-validation score: {random_search.best_score_:.4f}")
    log_message(f"Total time: {time.time() - start_time:.2f} seconds")
    
    return random_search.best_params_
