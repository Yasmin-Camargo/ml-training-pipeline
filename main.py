import os
import pandas as pd
from config import file_path, export_decision_tree_cpp, MAX_SAMPLES_PER_CLASS, is_validation_curve, REMOVE_COLUMNS, ACTIVE_GROUPINGS, CSV_SEPARATOR, RANDOM_STATE
from grouping import GROUPING_STRATEGIES
from logger import log_message
from grouping import *
from data_prep import balance_data, split_train_test
from feature_selection import recursive_feature_elimination_cv
from model_tuning import random_search
from validation_curves import generate_validation_curves
from sklearn.utils import resample
from sklearn.metrics import classification_report
from model_config import BASE_MODEL_CONFIG

def run_all_experiments():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep=CSV_SEPARATOR, low_memory=False)
    df = df.sample(n=5000, random_state=42) # TODO: remover depois

    # Remover colunas constantes
    """nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        log_message(f"Removed constant columns: {constant_cols}")
"""
    # Remover colunas especificadas
    if REMOVE_COLUMNS:
        df = df.drop(columns=REMOVE_COLUMNS, errors='ignore')
        log_message(f"Removed columns: {REMOVE_COLUMNS}")
        
    # Identificar e remover linhas com valores nulos
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    total_null_rows = df.isnull().any(axis=1).sum()
    if total_null_rows > 0:
        log_message(f"Found nulls in columns: {cols_with_nulls.to_dict()}")
        log_message(f"Removed {total_null_rows} rows containing null values.")
        df = df.dropna(how='any')
    else:
        log_message("No null values found.")

    # Remover linhas duplicadas
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        log_message(f"Removed {before - after} duplicate rows.")

    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].clip(lower=-1e18, upper=1e18)

    for grouping_name in ACTIVE_GROUPINGS:        
        try:
            df, groups = apply_grouping(df, grouping_name, GROUPING_STRATEGIES)
        except Exception as e:
            log_message(f"⚠ Warning: {e}. Skipping.")
            continue

        for model_id, model_config in MODEL_DEFINITIONS.items():
            
            process_fn = model_config['process_function']
            classifier_name = model_config['classifier_name']
            
            log_message(f"--- Preparing data for Model {model_id} ({classifier_name}) ---")
            
            try:
                df_model = process_fn(df)
            except Exception as e:
                log_message(f"Error preparing data for model {model_id}: {e}")
                continue
            
            for block_group in groups:
                try:
                    log_message(f"--- Processing ({grouping_name}) | Model {model_id} | Group {block_group} ---")
                    
                    df_balance = balance_data(df_model, block_group)
                    
                    X_train, X_test, y_train, y_test = split_train_test(df_balance)

                    df_train_temp = pd.concat([X_train, y_train], axis=1)
                    df_train_sampled = pd.concat([
                        resample(g, replace=False, n_samples=min(len(g), MAX_SAMPLES_PER_CLASS), random_state=RANDOM_STATE)
                        for _, g in df_train_temp.groupby(y_train.name)
                    ])
                    X_train_sampled = df_train_sampled.drop(columns=[y_train.name])
                    y_train_sampled = df_train_sampled[y_train.name]

                    if is_validation_curve:
                        generate_validation_curves(X_train_sampled, y_train_sampled, grouping_name, model_id, block_group)
                        continue

                    selected_mask = recursive_feature_elimination_cv(X_train_sampled, y_train_sampled, classifier_name)
                    selected_features = X_train.columns[selected_mask]
                    
                    # 2. Random Search alinhado (passando o nome)
                    best_params = random_search(X_train_sampled[selected_features], y_train_sampled, classifier_name)
                    
                    base_cfg = BASE_MODEL_CONFIG[classifier_name]
                    final_model = base_cfg['estimator'](**best_params)
                      
                    log_message(f"Training final {classifier_name} model...")
                    final_model.fit(X_train[selected_features], y_train)

                    final_preds = final_model.predict(X_test[selected_features])
                    #final_acc = accuracy_score(y_test, final_preds)
                    
                    final_report = classification_report(y_test, final_preds, zero_division=0)
                    log_message(f"Classification Report:\n{final_report}")
                    results_dir = f'results/{grouping_name}'
                    os.makedirs(results_dir, exist_ok=True)
                    filename = f"Result_{grouping_name}_model{model_id}_{block_group.replace('×', 'x')}.txt"
                    with open(os.path.join(results_dir, filename), 'w') as f:
                        f.write(final_report)

                    if classifier_name == 'decision_tree' and export_decision_tree_cpp:
                        export_tree_to_cpp(final_model, list(selected_features), sorted(y_train.unique()), 
                                           f"tree_{grouping_name}_m{model_id}_{block_group.replace('x', '_')}", 
                                           f'cpp_exports/{grouping_name}')

                except Exception as e:
                    log_message(f"✖ ERROR processing ({grouping_name}) Model {model_id}, Group {block_group}: {e}")

def export_tree_to_cpp(model, feature_names, class_names, function_name, output_dir):
    try:
        import DecisionTreeToCpp as to_cpp
        os.makedirs(output_dir, exist_ok=True)
        to_cpp.save_code(model, feature_names, class_names, function_name=function_name, output_dir=output_dir)
        log_message(f"✓ Model exported to C++ at: {output_dir}")
    except ImportError:
        log_message(f"⚠ DecisionTreeToCpp module not found. C++ export skipped.")
    except Exception as e:
        log_message(f"✖ Error exporting to C++: {e}")


if __name__ == "__main__":
    run_all_experiments()
