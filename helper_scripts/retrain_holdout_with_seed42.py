#!/usr/bin/env python3
"""
Script to re-evaluate holdout results with random_seed=42 or random_state=42.

Reads holdout_results.txt files from optimized experiment folders,
extracts the best parameters from each fold, adds random seed,
re-trains models on 80% data with different balancing strategies,
evaluates on 20% holdout, and saves to holdout_results42.txt
"""

import sys
import os
sys.path.append('/Users/i583975/git/tcc')

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import cycle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, confusion_matrix, auc
)
from sklearn.preprocessing import label_binarize

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.models import balance_fold
from src.process_data import load_hgnc_mapping

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configure matplotlib for better quality (matching plot_curves.py)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 22

# Standard color palette for all models (rainbow colors)
STANDARD_COLORS = {
    'decisiontree': '#FF0000',           # Red
    'randomforest': '#FF7F00',           # Orange
    'gradientboosting': '#FFFF00',       # Yellow
    'histogramgradientboosting': '#00FF00',  # Green
    'knearestneighbors': '#0000FF',     # Blue
    'multilayerperceptron': '#4B0082',  # Indigo
    'supportvectorclassifier': '#9400D3',  # Violet
    'catboost': '#FF1493',                # Pink
    'xgboost': '#00CED1',                 # Turquoise
}

# Additional colors for multiclass plots
MULTICLASS_COLORS = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3', '#FF1493']

# Model name display mapping
MODEL_DISPLAY_NAMES = {
    'catboost': 'CatBoost',
    'decision_tree': 'Decision Tree',
    'gradient_boosting': 'Gradient Boosting',
    'histogram_gradient_boosting': 'Histogram Gradient Boosting',
    'hist_gradient_boosting': 'Histogram Gradient Boosting',
    'k_nearest_neighbors': 'K-Nearest Neighbors',
    'knn': 'K-Nearest Neighbors',
    'multi_layer_perceptron': 'Multi-Layer Perceptron',
    'mlp': 'Multi-Layer Perceptron',
    'random_forest': 'Random Forest',
    'support_vector_classifier': 'Support Vector Classifier',
    'svc': 'Support Vector Classifier',
    'xgboost': 'XGBoost',
}

# Model class mapping
MODEL_CLASSES = {
    'catboost': CatBoostClassifier,
    'xgboost': XGBClassifier,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'gradient_boosting': GradientBoostingClassifier,
    'histogram_gradient_boosting': HistGradientBoostingClassifier,
    'hist_gradient_boosting': HistGradientBoostingClassifier,
    'knn': KNeighborsClassifier,
    'k_nearest_neighbors': KNeighborsClassifier,
    'mlp': MLPClassifier,
    'multi_layer_perceptron': MLPClassifier,
    'svc': SVC,
}


def format_model_name(model_name):
    """Format model name for display"""
    clean_name = model_name.lower().replace('_', '').replace('-', '').replace('metrics', '')
    if model_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_name]
    return model_name.replace('_', ' ').title()


def load_dataset(classification_type='binary'):
    """Load and prepare dataset with HGNC mapping"""
    features_path = "./data/UNION_features.tsv"
    labels_path = "./data/processed/UNION_labels.tsv"
    
    print(f"\nLoading data for {classification_type} classification...")
    print(f"  Features: {features_path}")
    print(f"  Labels: {labels_path}")
    
    # Load HGNC mapping
    hgnc_mapping, unmatched_genes, withdrawn_genes = load_hgnc_mapping()
    print(f"  HGNC mapping loaded: {len(hgnc_mapping)} mappings")
    
    # Load features
    features_df = pd.read_csv(features_path, sep='\t', index_col=0)
    original_feature_genes = len(features_df)
    
    # Load labels
    labels_df = pd.read_csv(labels_path, sep='\t', index_col=0)
    original_label_genes = len(labels_df)
    
    # Apply HGNC mapping to gene names (index) - map where available, keep original otherwise
    # This updates aliases/previous symbols to current approved symbols
    features_df.index = features_df.index.map(lambda x: hgnc_mapping.get(x, x))
    labels_df.index = labels_df.index.map(lambda x: hgnc_mapping.get(x, x))
    
    # Only remove genes that are explicitly unmatched or withdrawn
    unmatched_symbols = {g['input'] for g in unmatched_genes}
    withdrawn_symbols = {g['input'] for g in withdrawn_genes}
    genes_to_remove = unmatched_symbols | withdrawn_symbols
    
    features_df = features_df[~features_df.index.isin(genes_to_remove)]
    labels_df = labels_df[~labels_df.index.isin(genes_to_remove)]
    
    print(f"  Features after HGNC filtering: {original_feature_genes} → {len(features_df)} genes (removed {original_feature_genes - len(features_df)})")
    print(f"  Labels after HGNC filtering: {original_label_genes} → {len(labels_df)} genes (removed {original_label_genes - len(labels_df)})")
    
    # Find common genes
    common_genes = features_df.index.intersection(labels_df.index)
    print(f"  Common genes: {len(common_genes)}")
    
    # Filter to common genes
    labels_df = labels_df.loc[common_genes]
    features_df = features_df.loc[common_genes]
    
    # Select label column
    label_column = '3class' if classification_type == 'multiclass' else '2class'
    
    # Filter labeled data - need to filter both DataFrames together
    labeled_mask = labels_df[label_column].notna()
    labels_df_filtered = labels_df[labeled_mask]
    features_df_filtered = features_df.loc[labels_df_filtered.index]
    
    X = features_df_filtered.values
    y = labels_df_filtered[label_column].values.astype(int)
    gene_names = features_df_filtered.index.values
    
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y, gene_names


def parse_holdout_results_file(filepath):
    """Parse holdout_results.txt and extract fold parameters"""
    folds_data = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by fold sections
    fold_sections = content.split('=' * 80 + '\nFOLD ')
    
    for section in fold_sections[1:]:  # Skip first empty section
        lines = section.split('\n')
        
        # Extract fold number from first line
        fold_num = int(lines[0].split('\n')[0].strip())
        
        # Find the line with "Best Parameters:"
        for i, line in enumerate(lines):
            if line.startswith('Best Parameters:'):
                # Extract the dictionary from the line
                params_str = line.replace('Best Parameters:', '').strip()
                # Convert Python dict string to actual dict
                import ast
                params = ast.literal_eval(params_str)
                
                folds_data.append({
                    'fold': fold_num,
                    'params': params
                })
                break
    
    return folds_data


def add_random_seed_to_params(params, model_name):
    """Add random_seed or random_state to parameters based on model type"""
    params_copy = params.copy()
    
    # CatBoost uses random_seed, others use random_state
    if 'catboost' in model_name.lower():
        params_copy['random_seed'] = RANDOM_SEED
    else:
        # KNN doesn't need random_state (deterministic)
        if 'knn' not in model_name.lower() and 'k_nearest_neighbors' not in model_name.lower():
            params_copy['random_state'] = RANDOM_SEED
    
    return params_copy


def calculate_metrics(y_true, y_pred, y_pred_proba, classification_type):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    
    average_type = 'macro' if classification_type == "multiclass" else 'binary'
    precision = precision_score(y_true, y_pred, average=average_type, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_type, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_type, zero_division=0)
    
    if classification_type == "multiclass":
        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        pr_auc = average_precision_score(y_true, y_pred_proba, average='macro')
    else:
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        pr_auc = average_precision_score(y_true, y_pred_proba[:, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def evaluate_fold_with_balancing(fold_data, model_class, model_name, X_train, X_test, y_train, y_test, classification_type, balancing_strategies):
    """Evaluate a fold's parameters with different balancing strategies"""
    fold_num = fold_data['fold']
    params = add_random_seed_to_params(fold_data['params'], model_name)
    
    print(f"\nEvaluating Fold {fold_num} parameters...")
    
    results = {
        'fold': fold_num,
        'params': params,
        'balancing_results': {}
    }
    
    for balance_strat in balancing_strategies:
        print(f"  - Testing with balance strategy: {balance_strat}")
        
        # Apply balancing to training data
        if balance_strat == 'none':
            X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = balance_fold(X_train, y_train, balance_strat)
        
        # Create and train pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model_class(**params))
        ])
        
        pipeline.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate on holdout set
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba, classification_type)
        
        # Store predictions for plotting
        metrics['y_true'] = y_test
        metrics['y_pred'] = y_pred
        metrics['y_pred_proba'] = y_pred_proba
        
        results['balancing_results'][balance_strat] = metrics
    
    return results


def format_results_to_text(model_name, all_results, classification_type):
    """Format results in the same style as original holdout_results.txt"""
    lines = []
    
    lines.append(f"Holdout Evaluation Results for {model_name} (with random_seed=42)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Results for each fold's best parameters on holdout set")
    lines.append("Testing with different balancing strategies on training data (80%)")
    lines.append("Evaluated on unbalanced holdout set (20%)")
    lines.append("NOTE: All models trained with random_seed=42 or random_state=42")
    lines.append("-" * 80)
    lines.append("")
    
    # Results for each fold
    for result in all_results:
        lines.append("=" * 80)
        lines.append(f"FOLD {result['fold']}")
        lines.append("=" * 80)
        lines.append(f"Best Parameters: {result['params']}")
        lines.append("")
        lines.append("")
        
        for balance_strat, metrics in result['balancing_results'].items():
            lines.append("-" * 80)
            lines.append(f"BALANCING STRATEGY: {balance_strat.upper()}")
            lines.append("-" * 80)
            lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
            lines.append(f"  Precision: {metrics['precision']:.4f}")
            lines.append(f"  Recall:    {metrics['recall']:.4f}")
            lines.append(f"  F1-Score:  {metrics['f1']:.4f}")
            lines.append(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            lines.append(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
            lines.append("")
    
    # Find best combination by PR-AUC
    best_combinations = []
    for result in all_results:
        for balance_strat, metrics in result['balancing_results'].items():
            best_combinations.append({
                'fold': result['fold'],
                'balancing': balance_strat,
                'params': result['params'],
                'pr_auc': metrics['pr_auc'],
                'roc_auc': metrics['roc_auc'],
                'f1': metrics['f1'],
                'metrics': metrics
            })
    
    best_combinations.sort(key=lambda x: x['pr_auc'], reverse=True)
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("BEST PERFORMING COMBINATIONS ON HOLDOUT SET (by PR-AUC)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Top 5 combinations:")
    lines.append("-" * 80)
    lines.append("")
    
    for i, combo in enumerate(best_combinations[:5], 1):
        lines.append(f"{i}. Fold {combo['fold']} + {combo['balancing'].upper()}")
        lines.append(f"   PR-AUC: {combo['pr_auc']:.4f} | ROC-AUC: {combo['roc_auc']:.4f} | F1: {combo['f1']:.4f}")
        lines.append("")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("OVERALL BEST COMBINATION:")
    lines.append("=" * 80)
    best = best_combinations[0]
    lines.append(f"Fold: {best['fold']}")
    lines.append(f"Balancing Strategy: {best['balancing'].upper()}")
    lines.append(f"Parameters: {best['params']}")
    lines.append("")
    lines.append("Performance:")
    lines.append(f"  Accuracy:  {best['metrics']['accuracy']:.4f}")
    lines.append(f"  Precision: {best['metrics']['precision']:.4f}")
    lines.append(f"  Recall:    {best['metrics']['recall']:.4f}")
    lines.append(f"  F1-Score:  {best['metrics']['f1']:.4f}")
    lines.append(f"  ROC-AUC:   {best['metrics']['roc_auc']:.4f}")
    lines.append(f"  PR-AUC:    {best['metrics']['pr_auc']:.4f}")
    
    return '\n'.join(lines)


def plot_binary_curves(all_results, model_name, output_dir):
    """Plot ROC, PR curves and confusion matrix for binary classification (matching plot_curves.py style)"""
    output_dir = Path(output_dir) / 'curves'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best combination by PR-AUC
    best_result = None
    best_pr_auc = -1
    best_balance = None
    
    for result in all_results:
        for balance_strat, data in result['balancing_results'].items():
            if data['pr_auc'] > best_pr_auc:
                best_pr_auc = data['pr_auc']
                best_result = result
                best_balance = balance_strat
    
    # Get predictions for best combination
    best_data = best_result['balancing_results'][best_balance]
    y_true = best_data['y_true']
    y_pred = best_data['y_pred']
    y_pred_proba = best_data['y_pred_proba'][:, 1]
    
    # Get color and display name
    model_key = model_name.lower().replace('_', '').replace('-', '')
    color = STANDARD_COLORS.get(model_key, '#000000')
    model_display_name = format_model_name(model_name)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc_val = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 9))
    plt.plot(fpr, tpr, color=color, lw=3, 
            label=f'{model_display_name} (AUC = {roc_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8, 
             label='Random Classifier (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    plt.legend(loc="lower right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name}_roc_curve.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc_val = auc(recall, precision)
    
    plt.figure(figsize=(12, 9))
    plt.plot(recall, precision, color=color, lw=3, 
            label=f'{model_display_name} (AUC = {pr_auc_val:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.legend(loc="upper right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name}_pr_curve.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Passenger', 'Cancer Genes']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 22})
    plt.ylabel('True Label', fontsize=22)
    plt.xlabel('Predicted Label', fontsize=22)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name}_confusion_matrix.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"    Saved binary curves to {output_dir}")


def plot_multiclass_curves(all_results, model_name, output_dir):
    """Plot macro and OvR ROC/PR curves for multiclass classification (matching plot_curves_multiclass.py style)"""
    output_dir = Path(output_dir) / 'curves'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best combination by PR-AUC
    best_result = None
    best_pr_auc = -1
    best_balance = None
    
    for result in all_results:
        for balance_strat, data in result['balancing_results'].items():
            if data['pr_auc'] > best_pr_auc:
                best_pr_auc = data['pr_auc']
                best_result = result
                best_balance = balance_strat
    
    # Get predictions for best combination
    best_data = best_result['balancing_results'][best_balance]
    y_true = best_data['y_true']
    y_pred_proba = best_data['y_pred_proba']
    
    n_classes = 3
    class_names = ['Passenger', 'TSG', 'Oncogenes']
    
    # Get color and display name
    model_key = model_name.lower().replace('_', '').replace('-', '')
    color = STANDARD_COLORS.get(model_key, '#000000')
    model_display_name = format_model_name(model_name)
    
    # Binarize labels for OvR
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    # Compute ROC curve and ROC area for each class (OvR)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot macro ROC curve
    plt.figure(figsize=(12, 9))
    plt.plot(fpr["macro"], tpr["macro"], color=color, lw=3,
             label=f'{model_display_name} Macro-avg (AUC = {roc_auc["macro"]:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    plt.legend(loc="lower right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_roc_macro.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name}_roc_macro.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot OvR ROC curves
    plt.figure(figsize=(12, 9))
    colors = cycle(MULTICLASS_COLORS)
    for i in range(n_classes):
        color_class = next(colors)
        plt.plot(fpr[i], tpr[i], color=color_class, lw=3,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    plt.legend(loc="lower right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_roc_ovr.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name}_roc_ovr.pdf', bbox_inches='tight')
    plt.close()
    
    # Compute PR curve and PR area for each class (OvR)
    precision_dict = dict()
    recall_dict = dict()
    pr_auc = dict()
    
    for i in range(n_classes):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i])
        pr_auc[i] = auc(recall_dict[i], precision_dict[i])
    
    # Compute macro-average PR
    pr_auc["macro"] = np.mean([pr_auc[i] for i in range(n_classes)])
    
    # Plot macro PR (as average of class PRs)
    plt.figure(figsize=(12, 9))
    all_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        mean_precision += np.interp(all_recall, recall_dict[i][::-1], precision_dict[i][::-1])
    mean_precision /= n_classes
    
    plt.plot(all_recall, mean_precision, color=color, lw=3,
             label=f'{model_display_name} Macro-avg (AUC = {pr_auc["macro"]:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.legend(loc="upper right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pr_macro.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name}_pr_macro.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot OvR PR curves
    plt.figure(figsize=(12, 9))
    colors = cycle(MULTICLASS_COLORS)
    for i in range(n_classes):
        color_class = next(colors)
        plt.plot(recall_dict[i], precision_dict[i], color=color_class, lw=3,
                label=f'{class_names[i]} (AUC = {pr_auc[i]:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.legend(loc="upper right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pr_ovr.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name}_pr_ovr.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    y_pred = best_data['y_pred']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 22})
    plt.ylabel('True Label', fontsize=22)
    plt.xlabel('Predicted Label', fontsize=22)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name}_confusion_matrix.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"    Saved multiclass curves to {output_dir}")


def process_experiment_folder(experiment_path):
    """Process a single experiment folder (e.g., _optimized_binary_...)"""
    experiment_path = Path(experiment_path)
    print(f"\n{'='*80}")
    print(f"Processing experiment: {experiment_path.name}")
    print(f"{'='*80}")
    
    # Determine classification type from folder name
    if 'binary' in experiment_path.name.lower():
        classification_type = 'binary'
    elif 'multiclass' in experiment_path.name.lower():
        classification_type = 'multiclass'
    else:
        print(f"Cannot determine classification type from {experiment_path.name}")
        return
    
    # Load data
    print(f"Loading {classification_type} data...")
    X, y, gene_names = load_dataset(classification_type)
    
    # Split data 80/20 with same random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Balancing strategies to test
    balancing_strategies = ['none', 'randomundersampler', 'smoteenn', 'tomeklinks']
    
    # Process each model folder inside the experiment
    for model_folder in sorted(experiment_path.iterdir()):
        if not model_folder.is_dir():
            continue
        
        model_name = model_folder.name
        holdout_file = model_folder / 'holdout_results.txt'
        
        if not holdout_file.exists():
            print(f"  Skipping {model_name}: no holdout_results.txt found")
            continue
        
        print(f"\n  Processing model: {model_name}")
        
        # Parse the holdout results file
        try:
            folds_data = parse_holdout_results_file(holdout_file)
            print(f"    Found {len(folds_data)} folds")
        except Exception as e:
            print(f"    Error parsing holdout_results.txt: {e}")
            continue
        
        # Get model class
        model_class = MODEL_CLASSES.get(model_name)
        if model_class is None:
            print(f"    Unknown model class: {model_name}")
            continue
        
        # Evaluate each fold with different balancing strategies
        all_results = []
        for fold_data in folds_data:
            try:
                result = evaluate_fold_with_balancing(
                    fold_data, model_class, model_name,
                    X_train, X_test, y_train, y_test,
                    classification_type, balancing_strategies
                )
                all_results.append(result)
            except Exception as e:
                print(f"    Error evaluating fold {fold_data['fold']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save results to holdout_results42.txt
        if all_results:
            output_file = model_folder / 'holdout_results42.txt'
            output_text = format_results_to_text(model_name, all_results, classification_type)
            
            with open(output_file, 'w') as f:
                f.write(output_text)
            
            print(f"    Saved results to {output_file}")
            
            # Generate plots
            print(f"    Generating plots...")
            if classification_type == 'binary':
                plot_binary_curves(all_results, model_name, model_folder)
            else:
                plot_multiclass_curves(all_results, model_name, model_folder)


def main():
    """Main function to process all experiment folders"""
    results_base_path = Path('/Users/i583975/git/tcc/results_final_tcc')
    
    # Find all experiment folders that start with _optimized
    experiment_folders = [
        f for f in results_base_path.iterdir()
        if f.is_dir() and f.name.startswith('_optimized')
    ]
    
    print(f"Found {len(experiment_folders)} experiment folders to process:")
    for folder in sorted(experiment_folders):
        print(f"  - {folder.name}")
    
    # Process each experiment folder
    for experiment_folder in sorted(experiment_folders):
        try:
            process_experiment_folder(experiment_folder)
        except Exception as e:
            print(f"\nError processing {experiment_folder.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == '__main__':
    main()
