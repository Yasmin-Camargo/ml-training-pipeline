from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from .settings import ExperimentConfig

# Base model configurations
BASE_MODELS = {
    'decision_tree': {
        'estimator': DecisionTreeClassifier,
        'params': {
            'random_state': ExperimentConfig.RANDOM_STATE
        }
    },
    'logistic_regression': {
        'estimator': LogisticRegression,
        'params': {
            'random_state': ExperimentConfig.RANDOM_STATE,
            'max_iter': 10000
        }
    }
}

# Hyperparameter search spaces (Random Search)
SEARCH_SPACES = {
    'decision_tree': {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': list(range(2, 100, 10)),
        'min_samples_leaf': list(range(2, 200, 10)),
        'max_leaf_nodes': list(range(10, 500, 25)),
        'max_depth': list(range(1, 30)),
    },
    'logistic_regression': {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [1000],
    }
}


# Validation Curve Parameters
VAL_CURVE_PARAMS = {
    'decision_tree': {
        "max_depth": list(range(2, 10)) + list(range(10, 101, 10)),
        "min_samples_split": list(range(2, 25)) + list(range(25, 1001, 25)),
        "min_samples_leaf": list(range(1, 25)) + list(range(25, 1001, 25)),
        "max_leaf_nodes": list(range(2, 25)) + list(range(25, 1000, 25)),
    },
    'logistic_regression': {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'max_iter': [100, 200, 500, 1000, 10000]
    }
}