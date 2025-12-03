# ----------------------------------------------
#from config.settings_decision_mts import DataConfig, ExperimentConfig, RESULTS_DIR, EXPORTS_DIR
from config.settings_decision_intra import DataConfig, ExperimentConfig, RESULTS_DIR, EXPORTS_DIR
# ----------------------------------------------
   
import os
import sys
from sklearn.metrics import classification_report
from src.utils import log_message
from src.data import load_and_clean_data
from src.grouping import apply_grouping_strategy
from config.model_strategies import MODEL_STRATEGIES
from src.preprocessing import balance_group_data, split_and_sample, normalize_data
from src.feature_selection import run_rfe
from src.training import tune_hyperparameters, train_final_model
import src.DecisionTreeToCpp as to_cpp
from src.visualization import generate_validation_curves

def export_tree_to_cpp(model, feature_names, class_names, function_name, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        import src.DecisionTreeToCpp as to_cpp
        to_cpp.save_code(model, feature_names, class_names, function_name=function_name)
        src_file = function_name + '.h'
        dst_file = os.path.join(output_dir, src_file)
        shutil.move(src_file, dst_file)
        log_message(f"✓ Model exported to C++ at: {dst_file}")
    except ImportError:
        log_message(f"⚠ DecisionTreeToCpp module not found. C++ export skipped.")
    except Exception as e:
        log_message(f"✖ Error exporting to C++: {e}")

def main():
    log_message("=== Starting VVC ML Pipeline ===")
    
    # 1. Load Data
    try:
        df_raw = load_and_clean_data(DataConfig.FILE_PATH)
        #df_raw = df_raw.sample(n=5000, random_state=42) # TODO: remover depois
    except Exception as e:
        log_message(f"Critical Error loading data: {e}")
        sys.exit(1)

    # 2. Iterate over Grouping Strategies (area, max, single, etc.)
    for grouping_name in ExperimentConfig.ACTIVE_GROUPINGS:
        
        try:
            df_grouped, groups = apply_grouping_strategy(df_raw, grouping_name)
        except ValueError as e:
            log_message(f"Skipping strategy {grouping_name}: {e}")
            continue

        # 3. Iterate over MODEL_STRATEGIES (Model A, Model B)
        for model_strategie_id, model_strategie_cfg in MODEL_STRATEGIES.items():
            log_message(f"\n--- model_strategie: {model_strategie_id} ({model_strategie_cfg['description']}) ---")
            
            # Apply specific model_strategie transformation (label modification/filtering)
            df_model_strategie = model_strategie_cfg['process_function'](df_grouped)
            
            # Dynamic Model Type
            current_model_type = model_strategie_cfg['classifier_type']

            # 4. Iterate over Block Groups (e.g., 64x64, 32x32)
            for block_group in groups:
                group_id_clean = block_group.replace(":", "-").replace("×", "x")
                log_message(f"\n>>> Processing Group: {block_group} | Strategy: {grouping_name} | Model Type: {current_model_type}")
                
                # A. Filter by block group
                df_block = df_model_strategie[df_model_strategie['BlockGroup'] == block_group].copy()
                if df_block.empty:
                    continue

                # B. Balance Data
                df_balanced = balance_group_data(df_block)
                
                # C. Split Train/Test and Sample for Tuning
                X_train, X_test, y_train, y_test, X_train_samp, y_train_samp = split_and_sample(df_balanced)
                
                # C.1 Normalize Data if configured
                if ExperimentConfig.NORMALIZE_DATA:
                    X_train, X_test, X_train_samp = normalize_data(X_train, X_test, X_train_samp)
                
                # D. (Optional) Validation Curves
                if ExperimentConfig.RUN_VALIDATION_CURVES:
                    subdir = f"{grouping_name}_{model_strategie_id}_{group_id_clean}"
                    generate_validation_curves(X_train_samp, y_train_samp, subdir)
                    continue 

                # E. Feature Selection (RFE) using dynamic model type
                selected_cols = run_rfe(X_train_samp, y_train_samp, current_model_type)
                
                # F. Hyperparameter Tuning (Random Search)
                best_params = tune_hyperparameters(X_train_samp[selected_cols], y_train_samp, current_model_type)
                
                # G. Final Training (Full Train set, Selected Features)
                final_model = train_final_model(X_train[selected_cols], y_train, current_model_type, best_params)
                
                # H. Evaluation
                preds = final_model.predict(X_test[selected_cols])
                report = classification_report(y_test, preds, zero_division=0)
                log_message(f"Classification Report:\n{report}")
                
                # Save Report
                os.makedirs(RESULTS_DIR, exist_ok=True)
                report_file = RESULTS_DIR / grouping_name / f"Result_{model_strategie_id}_{block_group.replace('×', 'x')}.txt"
                os.makedirs(os.path.dirname(report_file), exist_ok=True)
                with open(report_file, "w") as f:
                    f.write(report)

                # I. Export to C++ (Only if it's a decision tree)
                    if ExperimentConfig.EXPORT_CPP and current_model_type == 'decision_tree':
                        try:
                            export_tree_to_cpp(
                                final_model,
                                list(selected_cols),
                                sorted(y_train.unique()),
                                f"tree_{grouping_name}_m{model_strategie_id}_{block_group.replace('x', '_')}",
                                f'cpp_exports/{grouping_name}'
                            )
                        except Exception as e:
                            log_message(f"✖ ERROR exporting to C++ ({grouping_name}) Model {model_strategie_id}, Group {block_group}: {e}")

if __name__ == "__main__":
    main()

# Function to export tree to C++
def export_tree_to_cpp(model, feature_names, class_names, function_name, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        to_cpp.save_code(model, feature_names, class_names, function_name=function_name, output_dir=output_dir)
        log_message(f"✓ Model exported to C++ at: {output_dir}")
    except ImportError:
        log_message(f"⚠ DecisionTreeToCpp module not found. C++ export skipped.")
    except Exception as e:
        log_message(f"✖ Error exporting to C++: {e}")