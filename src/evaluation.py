import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, accuracy_score
from config.settings_decision_intra import DataConfig, ExperimentConfig, RESULTS_DIR
from .utils import log_message
from .visualization import generate_learning_curve

def plot_roc_and_adjust_threshold(y_true, y_probs, group_name, model_type, output_dir):
    """
    Plots the ROC Curve and calculates the mathematically optimal threshold (Youden's J statistic),
    along with the default 0.5 threshold baseline. Generates a side-by-side Confusion Matrix plot
    with high DPI and large fonts for academic publishing.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    results_thresholds = {}
    
    # =========================================================================
    # PART 1: ROC CURVE PLOT
    # =========================================================================
    plt.figure(figsize=(12, 9))
    plt.plot(fpr, tpr, color='darkgray', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
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
    
    plt.plot(fpr[idx_default], tpr[idx_default], marker='s', markersize=14, color='blue', 
             label=f'Default ML (Thresh 0.5)\nRecall: {tpr[idx_default]*100:.1f}% | Acc: {acc_default*100:.1f}%')

    # 2. THE BEST MATHEMATICAL TRADE-OFF: YOUDEN'S J STATISTIC
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
        
        saved_blocks_youden = cm_youden[0, 0] 
        log_message(f"  -> COMPLEXITY REDUCTION: VVC will skip {saved_blocks_youden} blocks!", level="INFO")
        
    # Highlight the Youden point with a huge purple star
    plt.plot(fpr[idx_youden], tpr[idx_youden], marker='*', markersize=24, color='purple', 
             label=f'Optimal (Youden)\nThresh: {opt_thresh_youden:.3f} | Recall: {tpr[idx_youden]*100:.1f}% | Acc: {acc_youden*100:.1f}%')

    # Plot visual configurations for ROC
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('False Positive Rate (Reduces Time Saving)', fontsize=16)
    plt.ylabel('True Positive Rate / Recall (Protects BD-Rate)', fontsize=16)
    plt.title(f'ROC Analysis with Youden\'s Optimal Threshold - {model_type} ({group_name})', fontsize=18, pad=15)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    roc_path = os.path.join(output_dir, f'roc_tradeoff_{model_type}_{group_name}.png')
    plt.savefig(roc_path, bbox_inches='tight', dpi=300)
    plt.close()

    # =========================================================================
    # PART 2: SIDE-BY-SIDE CONFUSION MATRIX PLOT (BEFORE vs AFTER)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left Plot: Default ML (Threshold 0.5)
    disp_default = ConfusionMatrixDisplay(confusion_matrix=cm_default, display_labels=[0, 1])
    disp_default.plot(ax=axes[0], cmap=plt.cm.Blues, values_format='d')
    axes[0].set_title(f"Before: Default ML (Threshold 0.500)\nAcc: {acc_default*100:.2f}% | Recall: {tpr[idx_default]*100:.2f}%", fontsize=16)
    axes[0].set_xlabel('Predicted label', fontsize=15)
    axes[0].set_ylabel('True label', fontsize=15)
    axes[0].tick_params(axis='both', labelsize=14)
    
    for text in disp_default.text_.ravel():
        text.set_fontsize(16)
    
    # Right Plot: Optimal Youden Threshold
    disp_youden = ConfusionMatrixDisplay(confusion_matrix=cm_youden, display_labels=[0, 1])
    disp_youden.plot(ax=axes[1], cmap=plt.cm.Oranges, values_format='d')
    axes[1].set_title(f"After: Youden's Index (Threshold {opt_thresh_youden:.3f})\nAcc: {acc_youden*100:.2f}% | Recall: {tpr[idx_youden]*100:.2f}%", fontsize=16)
    axes[1].set_xlabel('Predicted label', fontsize=15)
    axes[1].set_ylabel('True label', fontsize=15)
    axes[1].tick_params(axis='both', labelsize=14)
    
    for text in disp_youden.text_.ravel():
        text.set_fontsize(16)
    
    plt.suptitle(f"Confusion Matrix Comparison: {model_type} ({group_name})", fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, f'cm_comparison_{model_type}_{group_name}.png')
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()

    return results_thresholds



def generate_shap_explanation(final_model, X_train_samp, output_dir, model_type):
    """
    Generates a professional SHAP Summary Plot (Beeswarm) showing the top 10 features.
    """
    try:
        import shap
        log_message("Generating SHAP Summary Plot...", level="INFO")
        
        explainer = shap.TreeExplainer(final_model)
        
        shap_values = explainer.shap_values(X_train_samp)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]     
       
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]
            
        plt.figure(figsize=(12, 10))
        
        shap.summary_plot(
            shap_values, 
            X_train_samp, 
            plot_type="dot", 
            max_display=10, 
            show=False
        )
        
        plt.title(f"SHAP Impact on Model Decision - Top 10 Features ({model_type})", fontsize=18, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_summary_{model_type}.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        log_message("SHAP Beeswarm Plot (Top 10) saved successfully.", level="INFO")
    except Exception as e:
        log_message(f"Could not generate SHAP plot: {e}", level="WARNING")


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
            if ExperimentConfig.RUN_ROC_ANALYSIS and len(unique_classes) == 2: # Only performs ROC for binary problems
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
        
    
    # 7. SHAP Explanability Analysis   
    if ExperimentConfig.RUN_SHAP_ANALYSIS:
        try:
            model_out_dir = os.path.join(RESULTS_DIR, grouping_name, str(model_strategie_id))
            
            X_train_filtered = X_train[selected_cols].sample(n=min(500, len(X_train)))
            
            generate_shap_explanation(
                final_model, 
                X_train_filtered, 
                model_out_dir, 
                current_model_type
            )
        except Exception as e:
            log_message(f"SHAP explanation skipped: {e}", level="ERROR")


    # 8. Export to C++ via callback (Handles both Tree and LR)
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
