from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from config import RANDOM_STATE

MODEL_HYPERPARAMS = {
    'decision_tree': {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': list(range(2, 100, 10)),
        'min_samples_leaf': list(range(2, 200, 10)),
        'max_leaf_nodes': list(range(10, 500, 25)),
        'max_depth': list(range(1, 30)),
    },
    'logistic_regression': {
        'penalty': ['l2', 'none'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [100, 200, 500, 1000],
    }
}

BASE_MODEL_CONFIG = {
    'decision_tree': {
        'estimator': DecisionTreeClassifier,
        'params': {
            'random_state': RANDOM_STATE
        }
    },
    'logistic_regression': {
        'estimator': LogisticRegression,
        'params': {
            'random_state': RANDOM_STATE,
            'max_iter': 20000
        }
    }
}

