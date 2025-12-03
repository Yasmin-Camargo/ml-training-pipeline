import time
from sklearn.model_selection import RandomizedSearchCV
from config.settings import ExperimentConfig
from config.model_hyperparameters import BASE_MODELS, SEARCH_SPACES
from .utils import log_message

def tune_hyperparameters(X, y, model_type):
    """Executes RandomizedSearchCV to find best params."""
    log_message(f"Starting Randomized Hyperparameter Search for {model_type}...")
    start_time = time.time()
    
    model_conf = BASE_MODELS[model_type]
    estimator = model_conf['estimator'](**model_conf['params'])
    
    if model_type not in SEARCH_SPACES:
        raise ValueError(f"Search space not defined for {model_type}")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=SEARCH_SPACES[model_type],
        n_iter=ExperimentConfig.RANDOM_SEARCH_ITER,
        cv=ExperimentConfig.CV_FOLDS,
        scoring=ExperimentConfig.SCORING,
        n_jobs=ExperimentConfig.N_JOBS,
        random_state=ExperimentConfig.RANDOM_STATE
    )
    
    search.fit(X, y)
    log_message(f"Best parameters found: {search.best_params_}")
    log_message(f"Best cross-validation score: {search.best_score_:.4f}")
    log_message(f"Total time: {time.time() - start_time:.2f} seconds")
    
    return search.best_params_

def train_final_model(X_train, y_train, model_type, best_params):
    """Trains the final model with best params on full training set."""
    log_message(f"Training final {model_type} model...")
    model_conf = BASE_MODELS[model_type]
    
    # Merge static params with tuned params
    final_params = {**model_conf['params'], **best_params}
    
    model = model_conf['estimator'](**final_params)
    model.fit(X_train, y_train)
    return model