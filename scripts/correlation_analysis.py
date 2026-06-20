import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure Python finds 'src' and 'config' when running from the /scripts directory
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from config.settings_decision_intra import DataConfig
from src.data import load_and_clean_data

def analyze_correlated_pairs(df, target_col, threshold=0.90):
    print(f"\n[ANALYSIS] Calculating correlation matrix (Threshold > {threshold})...")
    
    # 1. Keep only numeric columns for the calculation
    df_numeric = df.select_dtypes(include=[np.number])
    
    # 2. Calculate Pearson correlation matrix
    corr_matrix = df_numeric.corr()
    
    # 3. Calculate absolute correlation of all features with the Target
    target_corr = {}
    if target_col in df.columns:
        df_numeric_with_target = df[[target_col]].join(df_numeric.drop(columns=[target_col], errors='ignore'))
        full_corr = df_numeric_with_target.corr()
        if target_col in full_corr.columns:
            target_corr = full_corr[target_col].abs().to_dict()
    
    # 4. Find highly correlated pairs using only the upper triangle (avoids duplicating A-B and B-A)
    high_pairs = []
    columns = corr_matrix.columns
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_A = columns[i]
            col_B = columns[j]
            
            # Ignore the target itself in the redundant pairs list
            if col_A == target_col or col_B == target_col:
                continue
                
            val_corr = abs(corr_matrix.iloc[i, j])
            
            if val_corr >= threshold:
                high_pairs.append({
                    'Feature A': col_A,
                    'Feature B': col_B,
                    'Correlation': val_corr,
                    'Corr_A_Target': target_corr.get(col_A, 0.0),
                    'Corr_B_Target': target_corr.get(col_B, 0.0)
                })
                
    # Sort the pairs from highest to lowest correlation
    high_pairs = sorted(high_pairs, key=lambda x: x['Correlation'], reverse=True)
    
    # 5. Generate and save the Heatmap visual in the project root
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr_matrix.abs(), annot=False, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Absolute Feature Correlation Matrix')
    plt.tight_layout()
    
    plot_path = os.path.join(root_path, 'correlation_matrix.png')
    plt.savefig(plot_path)
    print(f"[VISUAL] Heatmap successfully saved at: {plot_path}")
    
    return high_pairs

if __name__ == "__main__":
    print("Loading and cleaning data through the pipeline...")
    df = load_and_clean_data(DataConfig.FILE_PATH)
    
    target = DataConfig.TARGET_COLUMN
    defined_threshold = 0.90  
    
    pairs = analyze_correlated_pairs(df, target, threshold=defined_threshold)
    
    print("\n" + "=" * 110)
    print(f" REDUNDANCY REPORT (Absolute Correlation >= {defined_threshold})")
    print("=" * 110)
    
    if not pairs:
        print(f"Excellent! No features showed a correlation greater than or equal to {defined_threshold}.")
    else:
        print(f"Found {len(pairs)} highly redundant feature pairs.\n")
        print(f"{'Feature A':<30} x {'Feature B':<30} | {'Pair Corr':<10} | {'Corr A-Target':<13} | {'Corr B-Target':<13}")
        print("-" * 110)
        
        suggested_removals = set()
        
        for pair in pairs:
            f_A = pair['Feature A']
            f_B = pair['Feature B']
            c_pair = pair['Correlation']
            c_A_t = pair['Corr_A_Target']
            c_B_t = pair['Corr_B_Target']
            
            print(f"{f_A:<30} x {f_B:<30} | {c_pair:.4f}     | {c_A_t:.4f}          | {c_B_t:.4f}")
            
            # Suggestion logic based on lowest correlation with the target
            if c_A_t >= c_B_t:
                suggested_removals.add(f_B)
            else:
                suggested_removals.add(f_A)
                
        print("\n" + "=" * 110)
        print("[SUGGESTION] Features to consider adding to 'REMOVE_COLUMNS' based on Target correlation:")
        print("=" * 110)
        print(sorted(list(suggested_removals)))
        print("\n")