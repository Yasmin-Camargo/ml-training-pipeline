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
            'max_iter': 20000
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
    'logistic_regression': [
    {
        # Solvers that do NOT accept L1 or ElasticNet
        'solver': ['lbfgs', 'newton-cg', 'sag'],
        'penalty': ['l2', None], 
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [200, 500, 2000]
    },
    {
        # Solver that accepts L1 and ElasticNet
        'solver': ['saga'],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0, 0.5, 1],
        'max_iter': [200, 500, 2000]
    }
]
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
        "C": [0.01, 0.1, 1, 10, 100],
        "l1_ratio": [0, 0.5, 1]
    }
}