import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from config.settings_decision_mts import DataConfig, ExperimentConfig, RESULTS_DIR
from .utils import log_message
from .visualization import generate_learning_curve


def evaluate_and_save(
    final_model,
    X_test,
    y_test,
    X_train,
    y_train,
    selected_cols,
    grouping_name,
    model_strategie_id,
    block_group,
    current_model_type,
    best_params=None,
    export_tree_callback=None,
):
    """Run evaluation (report, confusion matrix, confidences), save report and
      optionally generate learning curve and export to C++ via callback."""

    log_message("--- Evaluation ---", level="stage")

    # Classification report
    try:
        preds = final_model.predict(X_test[selected_cols])
        report = classification_report(y_test, preds, zero_division=0)
        log_message(f"Classification Report:\n{report}", level="INFO")
    except Exception as e:
        log_message(f"Error producing classification report: {e}", level="ERROR")
        report = ""

    # Confusion matrix (logged, no plot)
    try:
        class_labels = sorted(list(y_test.unique()))
        cm = confusion_matrix(y_test, preds, labels=class_labels)
        log_message(f"Confusion Matrix Labels: {class_labels}", level="INFO")
        cm_str = np.array2string(cm, separator=', ')
        log_message(f"Confusion Matrix:\n{cm_str}", level="INFO")
    except Exception as e:
        log_message(f"Error computing confusion matrix: {e}", level="ERROR")

    # Probabilities / confidence
    if hasattr(final_model, "predict_proba"):
        try:
            probas = final_model.predict_proba(X_test[selected_cols])
            confidence_per_sample = np.max(probas, axis=1)

            mean_overall = np.mean(confidence_per_sample)
            log_message(f">>> Group {block_group} - Mean Confidence (Overall): {mean_overall*100:.2f}%", level="INFO")

            target_name = DataConfig.TARGET_COLUMN
            unique_classes = sorted(y_test.unique())
            for cls in unique_classes:
                mask = (y_test == cls)
                if mask.sum() > 0:
                    mean_cls = np.mean(confidence_per_sample[mask])
                    log_message(f"    - {target_name} = {cls}: {mean_cls*100:.2f}% mean confidence", level="INFO")
        except Exception as e:
            log_message(f"Error computing probabilities/confidence: {e}", level="ERROR")

    # Save textual report
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        report_file = RESULTS_DIR / grouping_name / f"Result_{model_strategie_id}_{block_group.replace('×', 'x')}.txt"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, "w") as f:
            f.write(report)
    except Exception as e:
        log_message(f"Error saving report file: {e}", level="ERROR")

    # Logistic regression parameters (for C++ export copy/paste)
    if current_model_type == 'logistic_regression':
        try:
            log_message(f"--- Logistic Regression Parameters (Save for C++) ---", level="INFO")
            log_message(f"Group: {block_group}", level="INFO")
            bias = final_model.intercept_[0]
            log_message(f"Bias (Intercept): {bias:.16f}", level="INFO")
            weights = final_model.coef_[0]
            cpp_weights = ", ".join([f"{w:.16f}" for w in weights])
            log_message(f"Weights: {{ {cpp_weights} }}", level="DEBUG")
            log_message(f"Features: {list(selected_cols)}", level="DEBUG")
        except Exception as e:
            log_message(f"Error logging logistic regression params: {e}", level="ERROR")

    # End-of-pipeline learning curve
    try:
        if ExperimentConfig.RUN_LEARNING_CURVES_AT_END:
            log_message(f"--- Generating learning curve after final training ---", level="stage")
            subdir_final = f"{grouping_name}_{model_strategie_id}_{block_group.replace('×', 'x')}_final"
            generate_learning_curve(
                X_train[selected_cols],
                y_train,
                subdir_final,
                model_type=current_model_type,
                train_sizes=ExperimentConfig.LEARNING_CURVE_TRAIN_SIZES,
                best_params=best_params,
            )
    except Exception as e:
        log_message(f"Error generating end-of-pipeline learning curve: {e}", level="ERROR")

    # Export to C++ via callback (keeps module decoupled from main)
    if ExperimentConfig.EXPORT_CPP and current_model_type == 'decision_tree':
        try:
            if export_tree_callback:
                func_name = f"tree_{grouping_name}_m{model_strategie_id}_{block_group.replace('x', '_')}"
                export_tree_callback(
                    final_model,
                    list(selected_cols),
                    sorted(y_train.unique()),
                    func_name,
                    f'cpp_exports/{grouping_name}'
                )
            else:
                log_message("No export_tree_callback provided; skipping C++ export.", level="WARNING")
        except Exception as e:
            log_message(f"Error exporting to C++ ({grouping_name}) Model {model_strategie_id}, Group {block_group}: {e}", level="ERROR")

    return report
