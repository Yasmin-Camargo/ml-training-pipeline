import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, accuracy_score
from config.settings_decision_intra import DataConfig, ExperimentConfig, RESULTS_DIR
from .utils import log_message
from .visualization import generate_learning_curve

def plot_roc_and_adjust_threshold(y_true, y_probs, group_name, model_type, output_dir):
    """
    Plots the ROC Curve and calculates the mathematically optimal threshold (Youden's J statistic),
    along with the default 0.5 threshold baseline.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    results_thresholds = {}
    
    plt.figure(figsize=(13, 10))
    plt.plot(fpr, tpr, color='darkgray', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    log_message(f"--- TRADE-OFF ANALYSIS (FOCUS ON COMPLEXITY REDUCTION VS QUALITY) ---", level="INFO")
    
    # 1. Default Machine Learning (Threshold = 0.5)
    idx_default = np.argmin(np.abs(thresholds - 0.5))
    y_pred_default = (y_probs >= 0.5).astype(int)
    cm_default = confusion_matrix(y_true, y_pred_default)
    acc_default = accuracy_score(y_true, y_pred_default)
    
    if cm_default.shape == (2,2):
        fn_def = cm_default[1, 0]
        fp_def = cm_default[0, 1]
        log_message(f"[Default ML (Threshold 0.5)]", level="INFO")
        log_message(f"  -> Recall: {tpr[idx_default]*100:.1f}% | Accuracy: {acc_default*100:.2f}%", level="INFO")
        log_message(f"  -> Lost Blocks (FN): {fn_def} | Wasted processed blocks (FP): {fp_def}", level="INFO")
    
    plt.plot(fpr[idx_default], tpr[idx_default], marker='s', markersize=12, color='blue', 
             label=f'Default ML (Thresh 0.5)\nRecall: {tpr[idx_default]*100:.1f}% | Acc: {acc_default*100:.1f}%')

    # =========================================================================
    # 2. THE BEST MATHEMATICAL TRADE-OFF: YOUDEN'S J STATISTIC
    # Maximizes true positive rate (Recall) and minimizes false positive rate simultaneously
    # =========================================================================
    youden_j = tpr - fpr
    idx_youden = np.argmax(youden_j)
    opt_thresh_youden = thresholds[idx_youden]
    results_thresholds["Optimal Balance (Youden)"] = opt_thresh_youden
    
    y_pred_youden = (y_probs >= opt_thresh_youden).astype(int)
    cm_youden = confusion_matrix(y_true, y_pred_youden)
    acc_youden = accuracy_score(y_true, y_pred_youden)
    
    if cm_youden.shape == (2,2):
        fn_youden = cm_youden[1, 0]
        fp_youden = cm_youden[0, 1]
        log_message(f"[Optimal Balance (Youden's Index)] - RECOMMENDED", level="INFO")
        log_message(f"  -> Threshold: {opt_thresh_youden:.4f} | Recall: {tpr[idx_youden]*100:.1f}% | Accuracy: {acc_youden*100:.2f}%", level="INFO")
        log_message(f"  -> Lost Blocks (FN): {fn_youden} | Wasted processed blocks (FP): {fp_youden}", level="INFO")
        
        # How much time are we saving by skipping blocks?
        saved_blocks_youden = cm_youden[0, 0] # True Negatives (Where the time reduction is!)
        log_message(f"  -> COMPLEXITY REDUCTION: VVC will skip {saved_blocks_youden} blocks!", level="INFO")
        
    # Highlight the Youden point with a huge purple star
    plt.plot(fpr[idx_youden], tpr[idx_youden], marker='*', markersize=20, color='purple', 
             label=f'Optimal (Youden)\nThresh: {opt_thresh_youden:.3f} | Recall: {tpr[idx_youden]*100:.1f}% | Acc: {acc_youden*100:.1f}%')

    # Plot visual configurations
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Reduces Time Saving)')
    plt.ylabel('True Positive Rate / Recall (Protects BD-Rate)')
    plt.title(f'ROC Analysis with Youden\'s Optimal Threshold - {model_type} ({group_name})')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    roc_path = os.path.join(output_dir, f'roc_tradeoff_{model_type}_{group_name}.png')
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()

    return results_thresholds

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
    groups_train=None,
    export_model_callback=None,
):
    """
    Run evaluation (report, confusion matrix, confidences), save report,
    optionally generate learning curve and export to C++ via callback.
    """

    log_message("--- Evaluation ---", level="stage")

    # 1. Classification report
    try:
        preds = final_model.predict(X_test[selected_cols])
        report = classification_report(y_test, preds, zero_division=0)
        log_message(f"Classification Report:\n{report}", level="INFO")
    except Exception as e:
        log_message(f"Error producing classification report: {e}", level="ERROR")
        report = ""

    # 2. Confusion matrix (logged, no plot)
    try:
        class_labels = sorted(list(y_test.unique()))
        cm = confusion_matrix(y_test, preds, labels=class_labels)
        log_message(f"Confusion Matrix Labels: {class_labels}", level="INFO")
        cm_str = np.array2string(cm, separator=', ')
        log_message(f"Confusion Matrix:\n{cm_str}", level="INFO")
    except Exception as e:
        log_message(f"Error computing confusion matrix: {e}", level="ERROR")

    block_group_str = str(block_group).replace('×', 'x')

    # 3. Probabilities / Confidence Analysis & ROC Curve
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
            
            # === START OF THRESHOLD ANALYSIS (VVC) ===
            if len(unique_classes) == 2: # Only performs ROC for binary problems
                y_probs_positive = probas[:, 1]
                
                # Create a specific folder for the charts based on the grouping strategy
                model_out_dir = os.path.join(RESULTS_DIR, grouping_name, str(model_strategie_id))
                os.makedirs(model_out_dir, exist_ok=True)
                
                best_thresholds = plot_roc_and_adjust_threshold(
                    y_test, 
                    y_probs_positive, 
                    block_group_str, 
                    current_model_type, 
                    model_out_dir
                )
            # === END OF ANALYSIS ===

        except Exception as e:
            log_message(f"Error computing probabilities/confidence/ROC: {e}", level="ERROR")
    
    # 4. Save textual report to file
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Ex: results/area/Result_dst7_dst7_64x64.txt
        report_file = RESULTS_DIR / grouping_name / f"Result_{model_strategie_id}_{block_group_str}.txt"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, "w") as f:
            f.write(report)
    except Exception as e:
        log_message(f"Error saving report file: {e}", level="ERROR")

    # 5. Logistic Regression Parameters (Log for manual check)
    if current_model_type == 'logistic_regression':
        try:
            log_message(f"--- Logistic Regression Parameters ---", level="INFO")
            log_message(f"Group: {block_group}", level="INFO")
            
            # Intercept (Bias)
            if hasattr(final_model, 'intercept_') and len(final_model.intercept_) > 0:
                bias = final_model.intercept_[0]
                log_message(f"Bias (Intercept): {bias:.16f}", level="INFO")
            
            # Coefficients (Weights)
            if hasattr(final_model, 'coef_') and len(final_model.coef_) > 0:
                weights = final_model.coef_[0]
                cpp_weights = ", ".join([f"{w:.16f}" for w in weights])
                log_message(f"Weights: {{ {cpp_weights} }}", level="DEBUG")
                log_message(f"Features: {list(selected_cols)}", level="DEBUG")
                
        except Exception as e:
            log_message(f"Error logging logistic regression params: {e}", level="ERROR")

    # 6. End-of-pipeline learning curve
    try:
        if ExperimentConfig.RUN_LEARNING_CURVES_AT_END:
            log_message(f"--- Generating learning curve after final training ---", level="stage")
            subdir_final = f"{grouping_name}_{model_strategie_id}_{block_group_str}_final"
            generate_learning_curve(
                X_train[selected_cols],
                y_train,
                subdir_final,
                model_type=current_model_type,
                train_sizes=ExperimentConfig.LEARNING_CURVE_TRAIN_SIZES,
                best_params=best_params,
                groups=groups_train,
            )
    except Exception as e:
        log_message(f"Error generating end-of-pipeline learning curve: {e}", level="ERROR")

    # 7. Export to C++ via callback (Handles both Tree and LR)
    if ExperimentConfig.EXPORT_CPP:
        try:
            if export_model_callback:
                func_name = f"{current_model_type}_{grouping_name}_m{model_strategie_id}_{block_group_str}"
                
                export_model_callback(
                    final_model,
                    list(selected_cols),
                    sorted(y_train.unique()),
                    func_name,
                    f'cpp_exports/{grouping_name}',
                    current_model_type
                )
            else:
                log_message("No export_model_callback provided; skipping C++ export.", level="WARNING")
        except Exception as e:
            log_message(f"Error exporting to C++ ({grouping_name}) Model {model_strategie_id}, Group {block_group}: {e}", level="ERROR")

    return report
