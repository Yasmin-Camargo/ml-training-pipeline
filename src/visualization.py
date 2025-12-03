import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
from config.settings import ExperimentConfig, RESULTS_DIR
from config.model_hyperparameters import VAL_CURVE_PARAMS
from .utils import log_message

def generate_validation_curves(X, y, output_subdir):
    log_message(f"Generating validation curves for {output_subdir}...")
    
    # Using DecisionTree as default probe for complexity
    clf = DecisionTreeClassifier(random_state=ExperimentConfig.RANDOM_STATE)
    
    save_path = RESULTS_DIR / "val_curves" / output_subdir
    os.makedirs(save_path, exist_ok=True)

    for param_name, param_range in VAL_CURVE_PARAMS.items():
        train_scores, test_scores = validation_curve(
            clf, X, y,
            param_name=param_name,
            param_range=param_range,
            scoring=ExperimentConfig.SCORING,
            cv=ExperimentConfig.CV_FOLDS,
            n_jobs=ExperimentConfig.N_JOBS
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, np.mean(train_scores, axis=1), label="Training", marker='o')
        plt.plot(param_range, np.mean(test_scores, axis=1), label="Validation (CV)", marker='o')
        plt.title(f"Validation Curve - {param_name}")
        plt.xlabel(param_name)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path / f"{param_name}.png")
        plt.close()
    
    log_message(f"Validation curves saved to {save_path}")