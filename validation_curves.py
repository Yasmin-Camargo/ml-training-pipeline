import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from logger import log_message
from config import RANDOM_STATE, VAL_CURVE_PARAMS, VAL_CURVE_SCORING, VAL_CURVE_CV

def plot_and_save_validation_curve(X, y, param_name, param_range, model_name, model_num, block_group, scoring='accuracy', cv=5):
    log_message(f"Starting validation curve — Model {model_num}, Group '{block_group}', Parameter '{param_name}'")

    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    train_scores, test_scores = validation_curve(
        clf, X, y,
        param_name=param_name,
        param_range=param_range,
        scoring=VAL_CURVE_SCORING,
        cv=VAL_CURVE_CV,
        n_jobs=-1
    )

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, np.mean(train_scores, axis=1), label="Training", marker='o')
    plt.plot(param_range, np.mean(test_scores, axis=1), label="Validation (CV)", marker='o')
    plt.title(f"Validation Curve - {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    output_dir = f'val_curves/{model_name}_model{model_num}_{block_group.replace("×", "x")}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{param_name}.png"))
    plt.close()

def generate_validation_curves(X_sampled, y_sampled, grouping, model, block_group):
    for param_name, param_range in VAL_CURVE_PARAMS.items():
        plot_and_save_validation_curve(X_sampled, y_sampled, param_name, param_range, grouping, model, block_group)
