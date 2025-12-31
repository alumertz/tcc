#!/usr/bin/env python3
"""
Script to train and test all models from results.json configuration.
Saves test predictions and metrics for plotting.
"""

import sys
import os
sys.path.append('/Users/i583975/git/tcc')

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.models import balance_fold

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
    'mlp': MLPClassifier,
    'svc': SVC,
    'support_vector_classifier': SVC,
}


def load_dataset(classification_type='binary'):
    """Load and prepare dataset"""
    features_path = "./data/UNION_features.tsv"
    labels_path = "./data/processed/UNION_labels.tsv"
    
    print(f"\nLoading data for {classification_type} classification...")
    print(f"  Features: {features_path}")
    print(f"  Labels: {labels_path}")
    
    # Load features
    features_df = pd.read_csv(features_path, sep='\t', index_col=0)
    
    # Load labels
    labels_df = pd.read_csv(labels_path, sep='\t', index_col=0)
    
    # Find common genes
    common_genes = features_df.index.intersection(labels_df.index)
    
    # Filter to common genes
    labels_df = labels_df.loc[common_genes]
    features_df = features_df.loc[common_genes]
    
    # Select label column
    label_column = '3class' if classification_type == 'multiclass' else '2class'
    
    # Filter labeled data
    labeled_mask = labels_df[label_column].notna()
    X = features_df[labeled_mask].values
    y = labels_df.loc[labeled_mask, label_column].values.astype(int)
    gene_names = features_df[labeled_mask].index.values
    
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y, gene_names


def calculate_metrics(y_true, y_pred, y_pred_proba, classification_type='binary'):
    """Calculate evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    if classification_type == 'binary':
        # Binary classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Get probabilities for positive class
        if len(y_pred_proba.shape) == 2:
            y_pred_proba_pos = y_pred_proba[:, 1]
        else:
            y_pred_proba_pos = y_pred_proba
        
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba_pos)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba_pos)
        
    else:
        # Multiclass metrics
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Multiclass AUC (One-vs-Rest)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = 0.0
        
        try:
            metrics['average_precision'] = average_precision_score(
                y_true, y_pred_proba, average='weighted'
            )
        except:
            metrics['average_precision'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def train_and_evaluate_model(config, X, y, gene_names):
    """Train and evaluate a single model configuration"""
    
    model_name = config['model']
    classification_type = config['type']
    balancing = config['balancing']
    hyperparameters = config['hyperparameters']
    
    print(f"\n{'='*80}")
    print(f"Training: {model_name.upper()} ({classification_type})")
    print(f"  Balancing: {balancing}")
    print(f"  Hyperparameters: {hyperparameters}")
    print(f"{'='*80}")
    
    # Split data (80/20 train/test)
    X_train, X_test, y_train, y_test, genes_train, genes_test = train_test_split(
        X, y, gene_names, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Apply balancing to training data
    if balancing and balancing != 'none':
        print(f"Applying {balancing} balancing...")
        X_train, y_train = balance_fold(X_train, y_train, balancing)
        print(f"After balancing: {X_train.shape[0]} samples")
    
    # Get model class
    model_class = MODEL_CLASSES.get(model_name)
    if model_class is None:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model_class(**hyperparameters))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Get predictions on test set
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, classification_type)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Average Precision (PR AUC): {metrics['average_precision']:.4f}")
    
    # Prepare results to save
    results = {
        'model_name': model_name,
        'classification_type': classification_type,
        'balancing': balancing,
        'hyperparameters': hyperparameters,
        'test_metrics': metrics,
        'test_predictions': {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'gene_names': genes_test.tolist()
        },
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return results


def main():
    """Main execution function"""
    
    print("="*80)
    print("TRAINING MODELS FROM results.json")
    print("="*80)
    
    # Load results.json
    results_json_path = "./results.json"
    print(f"\nLoading configuration from: {results_json_path}")
    
    with open(results_json_path, 'r') as f:
        configs = json.load(f)
    
    print(f"Found {len(configs)} model configurations")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results_final_tcc/trained_models_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Group configs by type
    binary_configs = [c for c in configs if c['type'] == 'binary']
    multiclass_configs = [c for c in configs if c['type'] == 'multiclass']
    
    print(f"\nBinary models: {len(binary_configs)}")
    print(f"Multiclass models: {len(multiclass_configs)}")
    
    # Train binary models
    if binary_configs:
        print(f"\n{'='*80}")
        print("TRAINING BINARY CLASSIFICATION MODELS")
        print(f"{'='*80}")
        
        X, y, gene_names = load_dataset('binary')
        
        binary_dir = os.path.join(output_dir, 'binary')
        os.makedirs(binary_dir, exist_ok=True)
        
        for config in binary_configs:
            try:
                results = train_and_evaluate_model(config, X, y, gene_names)
                
                if results:
                    # Save results
                    model_name = config['model']
                    output_file = os.path.join(binary_dir, f"metrics_{model_name}.json")
                    
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    print(f"✅ Saved: {output_file}")
            
            except Exception as e:
                print(f"❌ Error training {config['model']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Train multiclass models
    if multiclass_configs:
        print(f"\n{'='*80}")
        print("TRAINING MULTICLASS CLASSIFICATION MODELS")
        print(f"{'='*80}")
        
        X, y, gene_names = load_dataset('multiclass')
        
        multiclass_dir = os.path.join(output_dir, 'multiclass')
        os.makedirs(multiclass_dir, exist_ok=True)
        
        for config in multiclass_configs:
            try:
                results = train_and_evaluate_model(config, X, y, gene_names)
                
                if results:
                    # Save results
                    model_name = config['model']
                    output_file = os.path.join(multiclass_dir, f"metrics_{model_name}.json")
                    
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    print(f"✅ Saved: {output_file}")
            
            except Exception as e:
                print(f"❌ Error training {config['model']}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    main()
