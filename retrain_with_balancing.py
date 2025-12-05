#!/usr/bin/env python3
"""
Script to retrain models using best parameters from previous runs,
testing different balancing strategies on the 80% training data
and evaluating on the 20% holdout set.
"""

import sys
import os
import re
import ast
sys.path.append('/Users/i583975/git/tcc')

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.models import balance_fold, calculate_metrics, get_model_predictions, FoldMetrics
from src.reports import save_holdout_results


# Model class mapping
MODEL_CLASSES = {
    'catboost': CatBoostClassifier,
    'xgboost': XGBClassifier,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'gradient_boosting': GradientBoostingClassifier,
    'histogram_gradient_boosting': HistGradientBoostingClassifier,
    'k_nearest_neighbors': KNeighborsClassifier,
    'multi_layer_perceptron': MLPClassifier,
    'svc': SVC
}


def load_dataset_for_retraining(features_path, labels_path, classification_type='binary'):
    """
    Load and prepare dataset specifically for retraining script.
    Handles the 'Unnamed: 0' column name issue.
    """
    print(f"\n{'='*80}")
    print("LOADING DATASET:")
    print(f"  Features: {features_path}")
    print(f"  Labels: {labels_path}")
    print(f"  Classification type: {classification_type}")
    
    # Load features
    features_df = pd.read_csv(features_path, sep='\t')
    print(f"  Features loaded: {features_df.shape}")
    print(f"  Features columns (first 5): {list(features_df.columns[:5])}")
    
    # Rename first column to 'gene' if needed
    if features_df.columns[0] == 'Unnamed: 0':
        features_df = features_df.rename(columns={'Unnamed: 0': 'gene'})
        print(f"  Renamed 'Unnamed: 0' to 'gene'")
    
    if 'gene' not in features_df.columns:
        print(f"  ERROR: No 'gene' column found! Columns: {list(features_df.columns[:5])}")
        return None, None, None, None
    
    # Load labels
    labels_df = pd.read_csv(labels_path, sep='\t')
    print(f"  Labels loaded: {labels_df.shape}")
    print(f"  Labels columns: {list(labels_df.columns)}")
    
    # Select appropriate label column
    label_column = '2class' if classification_type == 'binary' else '3class'
    print(f"  Using label column: {label_column}")
    
    labels_clean = pd.DataFrame({
        'gene': labels_df['genes'],
        'label': labels_df[label_column]
    })
    
    print(f"  Labels before dropping NaN: {len(labels_clean)}")
    print(f"  Label value counts before NaN drop:\n{labels_clean['label'].value_counts(dropna=False).sort_index()}")
    
    # Drop NaN values
    labels_clean = labels_clean.dropna(subset=['label'])
    print(f"  Labels after dropping NaN: {len(labels_clean)}")
    
    labels_clean['label'] = labels_clean['label'].astype(int)
    print(f"  Label value counts after conversion:\n{labels_clean['label'].value_counts().sort_index()}")
    
    # Find common genes
    common_genes = set(features_df['gene']) & set(labels_clean['gene'])
    print(f"  Common genes: {len(common_genes)}")
    
    # Filter and align
    features_aligned = features_df[features_df['gene'].isin(common_genes)].sort_values('gene').reset_index(drop=True)
    labels_aligned = labels_clean[labels_clean['gene'].isin(common_genes)].sort_values('gene').reset_index(drop=True)
    
    print(f"  Features aligned: {features_aligned.shape}")
    print(f"  Labels aligned: {labels_aligned.shape}")
    
    # Extract X, y, gene_names
    X = features_aligned.drop('gene', axis=1).values
    y = labels_aligned['label'].values
    gene_names = features_aligned['gene'].values
    feature_names = features_aligned.drop('gene', axis=1).columns.tolist()
    
    print(f"\nFINAL DATASET:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Unique classes in y: {np.unique(y)}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"{'='*80}\n")
    
    return X, y, gene_names, feature_names


def parse_holdout_results_old_format(file_path):
    """
    Parse the old format holdout_results.txt file to extract best parameters per fold.
    
    Args:
        file_path: Path to the holdout_results.txt file
        
    Returns:
        List of dicts with fold info: [{'fold': 1, 'params': {...}}, ...]
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    fold_params = []
    
    # Pattern to match fold sections
    fold_pattern = r'Fold (\d+):\s+Best Parameters: ({[^}]+})'
    
    matches = re.finditer(fold_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        fold_num = int(match.group(1))
        params_str = match.group(2)
        
        # Parse the parameters dict
        try:
            params = ast.literal_eval(params_str)
            fold_params.append({
                'fold': fold_num,
                'params': params
            })
        except Exception as e:
            print(f"Warning: Could not parse parameters for fold {fold_num}: {e}")
            continue
    
    return fold_params


def parse_holdout_results_new_format(file_path):
    """
    Parse the new format holdout_results.txt file to extract best parameters per fold.
    This format has the parameters after "Best Parameters:" on a separate line.
    
    Args:
        file_path: Path to the holdout_results.txt file
        
    Returns:
        List of dicts with fold info: [{'fold': 1, 'params': {...}}, ...]
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    fold_params = []
    
    # Pattern to match fold sections - more flexible
    fold_pattern = r'FOLD (\d+)\s*={70,}\s*Best Parameters: ({[^}]+})'
    
    matches = re.finditer(fold_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        fold_num = int(match.group(1))
        params_str = match.group(2)
        
        # Parse the parameters dict
        try:
            params = ast.literal_eval(params_str)
            fold_params.append({
                'fold': fold_num,
                'params': params
            })
        except Exception as e:
            print(f"Warning: Could not parse parameters for fold {fold_num}: {e}")
            continue
    
    return fold_params


def retrain_with_balancing_strategies(
    model_name,
    classifier_class,
    fold_params_list,
    X, y,
    classification_type,
    data_source,
    balance_strategies=None,
    output_dir=None,
    use_less_params=False
):
    """
    Retrain models using best parameters from each fold with different balancing strategies.
    
    Args:
        model_name: Name of the model
        classifier_class: The model class to instantiate
        fold_params_list: List of dicts with fold number and parameters
        X, y: Full dataset
        classification_type: 'binary' or 'multiclass'
        data_source: 'ana' or 'renan'
        balance_strategies: List of balancing strategies to test
        output_dir: Directory to save results (if None, will auto-generate)
        use_less_params: Whether to use less params naming
    """
    if balance_strategies is None:
        balance_strategies = ['none', 'randomundersampler', 'smoteenn', 'tomeklinks']
    
    # Debug: Print data info before split
    print(f"\n{'='*80}")
    print("DATA BEFORE SPLIT:")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Unique classes: {np.unique(y)}")
    print(f"  y min: {y.min()}, y max: {y.max()}")
    print(f"  y dtype: {y.dtype}")
    print(f"  First 20 y values: {y[:20]}")
    print(f"{'='*80}")
    
    # Split data into 80% train, 20% holdout (same as original training)
    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  Training set (80%): {X_train.shape[0]} samples")
    print(f"  Holdout set (20%): {X_test.shape[0]} samples")
    print(f"  Testing {len(balance_strategies)} balancing strategies on each of {len(fold_params_list)} folds")
    
    holdout_results = []
    
    for fold_info in fold_params_list:
        fold_num = fold_info['fold']
        params = fold_info['params']
        
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold_num}")
        print(f"Parameters: {params}")
        print(f"{'='*80}")
        
        fold_holdout_results = {
            'fold': fold_num,
            'best_params': params,
            'balance_strategies': {}
        }
        
        # Test with each balancing strategy
        for balance_strat in balance_strategies:
            print(f"  Testing with balance strategy: {balance_strat}")
            
            try:
                # Apply balancing only to the 80% training data
                if balance_strat == 'none':
                    X_train_balanced, y_train_balanced = X_train, y_train
                else:
                    X_train_balanced, y_train_balanced = balance_fold(X_train, y_train, balance_strat)
                
                # Train model with best params from this fold on the balanced 80% training data
                final_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', classifier_class(**params))
                ])
                final_pipeline.fit(X_train_balanced, y_train_balanced)
                
                # Evaluate on holdout set (unbalanced 20%)
                y_pred_holdout, y_pred_proba_holdout = get_model_predictions(final_pipeline, X_test)
                holdout_metrics = calculate_metrics(y_test, y_pred_holdout, y_pred_proba_holdout, classification_type)
                
                fold_holdout_results['balance_strategies'][balance_strat] = holdout_metrics
                
                print(f"    Accuracy: {holdout_metrics.accuracy:.4f}, F1: {holdout_metrics.f1:.4f}, "
                      f"ROC-AUC: {holdout_metrics.roc_auc:.4f}, PR-AUC: {holdout_metrics.pr_auc:.4f}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                # Create zero metrics as fallback
                fold_holdout_results['balance_strategies'][balance_strat] = FoldMetrics(
                    accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.0, pr_auc=0.0
                )
        
        holdout_results.append(fold_holdout_results)
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    
    # Use custom output_dir if provided, otherwise use the save_holdout_results function
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to custom location
        holdout_results_path = os.path.join(output_dir, "holdout_results.txt")
        with open(holdout_results_path, 'w') as f:
            f.write(f"Holdout Evaluation Results for {model_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Results for each fold's best parameters on holdout set\n")
            f.write("Testing with different balancing strategies on training data (80%)\n")
            f.write("Evaluated on unbalanced holdout set (20%)\n")
            f.write("-"*80 + "\n")
            
            for hr in holdout_results:
                f.write(f"\n{'='*80}\n")
                f.write(f"FOLD {hr['fold']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Best Parameters: {hr['best_params']}\n\n")
                
                # Write results for each balancing strategy
                for balance_strat, metrics in hr['balance_strategies'].items():
                    f.write(f"\n{'-'*80}\n")
                    f.write(f"BALANCING STRATEGY: {balance_strat.upper()}\n")
                    f.write(f"{'-'*80}\n")
                    f.write(f"  Accuracy:  {metrics.accuracy:.4f}\n")
                    f.write(f"  Precision: {metrics.precision:.4f}\n")
                    f.write(f"  Recall:    {metrics.recall:.4f}\n")
                    f.write(f"  F1-Score:  {metrics.f1:.4f}\n")
                    f.write(f"  ROC-AUC:   {metrics.roc_auc:.4f}\n")
                    f.write(f"  PR-AUC:    {metrics.pr_auc:.4f}\n")
            
            # Find best performing combination
            f.write(f"\n\n{'='*80}\n")
            f.write(f"BEST PERFORMING COMBINATIONS ON HOLDOUT SET (by PR-AUC)\n")
            f.write(f"{'='*80}\n\n")
            
            # Collect all combinations
            all_combinations = []
            for hr in holdout_results:
                for balance_strat, metrics in hr['balance_strategies'].items():
                    all_combinations.append({
                        'fold': hr['fold'],
                        'balance_strategy': balance_strat,
                        'params': hr['best_params'],
                        'metrics': metrics
                    })
            
            # Sort by PR-AUC
            all_combinations.sort(key=lambda x: x['metrics'].pr_auc, reverse=True)
            
            # Write top 5 combinations
            f.write("Top 5 combinations:\n")
            f.write("-"*80 + "\n")
            for i, combo in enumerate(all_combinations[:5], 1):
                f.write(f"\n{i}. Fold {combo['fold']} + {combo['balance_strategy'].upper()}\n")
                f.write(f"   PR-AUC: {combo['metrics'].pr_auc:.4f} | ")
                f.write(f"ROC-AUC: {combo['metrics'].roc_auc:.4f} | ")
                f.write(f"F1: {combo['metrics'].f1:.4f}\n")
            
            # Write best overall
            best_combo = all_combinations[0]
            f.write(f"\n\n{'='*80}\n")
            f.write(f"OVERALL BEST COMBINATION:\n")
            f.write(f"{'='*80}\n")
            f.write(f"Fold: {best_combo['fold']}\n")
            f.write(f"Balancing Strategy: {best_combo['balance_strategy'].upper()}\n")
            f.write(f"Parameters: {best_combo['params']}\n\n")
            f.write(f"Performance:\n")
            f.write(f"  Accuracy:  {best_combo['metrics'].accuracy:.4f}\n")
            f.write(f"  Precision: {best_combo['metrics'].precision:.4f}\n")
            f.write(f"  Recall:    {best_combo['metrics'].recall:.4f}\n")
            f.write(f"  F1-Score:  {best_combo['metrics'].f1:.4f}\n")
            f.write(f"  ROC-AUC:   {best_combo['metrics'].roc_auc:.4f}\n")
            f.write(f"  PR-AUC:    {best_combo['metrics'].pr_auc:.4f}\n")
        
        print(f"Holdout evaluation results saved to: {holdout_results_path}")
    else:
        # Use the standard save function
        save_holdout_results(
            model_name, holdout_results,
            data_source, classification_type,
            balance_strategy='none',  # This is just for folder naming
            use_less_params=use_less_params
        )
    
    print("Done!")
    return holdout_results


def process_results_directory(results_dir, classification_type, data_source, use_less_params=False):
    """
    Process a results directory to retrain models with different balancing strategies.
    
    Args:
        results_dir: Path to the results directory (e.g., optimized_multiclass_less/...)
        classification_type: 'binary' or 'multiclass'
        data_source: 'ana' or 'renan'
        use_less_params: Whether to use less params naming
    """
    print(f"\n{'='*80}")
    print(f"Processing: {results_dir}")
    print(f"Classification type: {classification_type}")
    print(f"Data source: {data_source}")
    print(f"{'='*80}")
    
    # Prepare the dataset
    features_path = '/Users/i583975/git/tcc/data/UNION_features.tsv'
    labels_path = '/Users/i583975/git/tcc/data/processed/UNION_labels.tsv'
    
    X, y, gene_names, feature_names = load_dataset_for_retraining(features_path, labels_path, classification_type)
    
    if X is None:
        print("Error: Failed to load dataset")
        return
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Iterate through model directories
    for model_dir_name in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir_name)
        
        # Skip files, only process directories
        if not os.path.isdir(model_path):
            continue
        
        holdout_file = os.path.join(model_path, 'holdout_results.txt')
        
        # Skip if holdout_results.txt doesn't exist
        if not os.path.exists(holdout_file):
            print(f"Skipping {model_dir_name}: no holdout_results.txt found")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing model: {model_dir_name}")
        print(f"{'='*80}")
        
        # Check if file already has the new format
        with open(holdout_file, 'r') as f:
            content = f.read()
            if 'BALANCING STRATEGY:' in content:
                print(f"Skipping {model_dir_name}: already has new format with balancing strategies")
                continue
        
        # Parse the holdout results file
        # Try new format first
        fold_params = parse_holdout_results_new_format(holdout_file)
        
        # If no results, try old format
        if not fold_params:
            fold_params = parse_holdout_results_old_format(holdout_file)
        
        if not fold_params:
            print(f"Warning: Could not parse holdout_results.txt for {model_dir_name}")
            continue
        
        print(f"Found {len(fold_params)} folds with parameters")
        
        # Get model class
        classifier_class = MODEL_CLASSES.get(model_dir_name)
        if classifier_class is None:
            print(f"Warning: Unknown model type {model_dir_name}, skipping")
            continue
        
        # Retrain with balancing strategies
        try:
            retrain_with_balancing_strategies(
                model_name=model_dir_name,
                classifier_class=classifier_class,
                fold_params_list=fold_params,
                X=X, y=y,
                classification_type=classification_type,
                data_source=data_source,
                output_dir=model_path,  # Save to the same directory
                use_less_params=use_less_params
            )
        except Exception as e:
            print(f"Error processing {model_dir_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """Main function to process both multiclass and binary results"""
    
    # Process multiclass results
    multiclass_dir = '/Users/i583975/git/tcc/results/optimized_multiclass_less/20251127_044704_ana_optimized_multiclass_undersampler_less'
    print("\n" + "="*80)
    print("PROCESSING MULTICLASS RESULTS")
    print("="*80)
    process_results_directory(
        results_dir=multiclass_dir,
        classification_type='multiclass',
        data_source='ana',
        use_less_params=True
    )
    
    # Process binary results
    binary_dir = '/Users/i583975/git/tcc/results/optimized_binary_less/20251125_160305_ana_optimized_binary_rundersampler_less'
    print("\n\n" + "="*80)
    print("PROCESSING BINARY RESULTS")
    print("="*80)
    process_results_directory(
        results_dir=binary_dir,
        classification_type='binary',
        data_source='ana',
        use_less_params=True
    )
    
    print("\n\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
