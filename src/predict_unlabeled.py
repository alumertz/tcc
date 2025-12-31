#!/usr/bin/env python3
"""
Script to train a model on labeled data and predict on unlabeled data.
Edit the MODEL_CONFIG section below to specify your classifier and parameters.
"""

import sys
import os
sys.path.append('/Users/i583975/git/tcc')

import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.processing import prepare_dataset
from src.models import balance_fold

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# MODEL CONFIGURATION - EDIT THIS SECTION
# ============================================================================

# Configuration will be set based on USE_MULTICLASS below
# Binary: GradientBoostingClassifier with tomeklinks
# Multiclass: MLPClassifier with no balancing
MODEL_CONFIG = None  # Will be set dynamically based on USE_MULTICLASS

# Example configurations for different models:

# Data configuration
USE_MULTICLASS = False  # Set to True for multiclass, False for binary
USE_MLP = True  # Set to True to use MLP for multiclass, False for Gradient Boosting
# BALANCE_STRATEGY will be set automatically based on USE_MULTICLASS
# Binary: 'tomeklinks', Multiclass: 'none'
BALANCE_STRATEGY = None  # Will be set dynamically
OUTPUT_FILE = None  # Will be set dynamically based on USE_MULTICLASS

# ============================================================================
# END OF CONFIGURATION
# ============================================================================


def load_data(use_multiclass=False):
    """Load labeled and unlabeled data"""
    
    features_path = "./data/UNION_features.tsv"
    labels_path = "./data/processed/UNION_labels.tsv"
    
    print("Loading data...")
    print(f"  Features: {features_path}")
    print(f"  Labels: {labels_path}")
    
    # Load features
    features_df = pd.read_csv(features_path, sep='\t', index_col=0)
    
    # Load labels
    labels_df = pd.read_csv(labels_path, sep='\t', index_col=0)
    
    # Find genes that exist in both features and labels
    common_genes = features_df.index.intersection(labels_df.index)
    genes_only_in_labels = labels_df.index.difference(features_df.index)
    
    if len(genes_only_in_labels) > 0:
        print(f"\nWarning: {len(genes_only_in_labels)} genes in labels file not found in features file:")
        print(f"  {list(genes_only_in_labels[:10])}" + ("..." if len(genes_only_in_labels) > 10 else ""))
        print(f"  These genes will be skipped.")
    
    # Filter to only common genes
    labels_df = labels_df.loc[common_genes]
    features_df = features_df.loc[common_genes]
    
    # Determine classification type
    classification_type = 'multiclass' if use_multiclass else 'binary'
    label_column = '3class' if use_multiclass else '2class'
    
    print(f"\nClassification type: {classification_type}")
    print(f"Using label column: {label_column}")
    
    # Split into labeled and unlabeled
    labeled_mask = labels_df[label_column].notna()
    
    # Labeled data
    labeled_genes = labels_df[labeled_mask].index
    X_labeled = features_df.loc[labeled_genes].values
    y_labeled = labels_df.loc[labeled_genes, label_column].values
    
    # Unlabeled data
    unlabeled_genes = labels_df[~labeled_mask].index
    X_unlabeled = features_df.loc[unlabeled_genes].values
    
    print(f"\nData summary:")
    print(f"  Total genes (in both files): {len(labels_df)}")
    print(f"  Labeled genes: {len(labeled_genes)}")
    print(f"  Unlabeled genes: {len(unlabeled_genes)}")
    print(f"  Features: {X_labeled.shape[1]}")
    
    if classification_type == 'binary':
        print(f"\nClass distribution (labeled):")
        print(f"  Class 0: {np.sum(y_labeled == 0)}")
        print(f"  Class 1: {np.sum(y_labeled == 1)}")
    else:
        print(f"\nClass distribution (labeled):")
        unique, counts = np.unique(y_labeled, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {int(cls)}: {count}")
    
    return X_labeled, y_labeled, X_unlabeled, unlabeled_genes, labeled_genes


def train_and_predict():
    """Train model on labeled data and predict on unlabeled data"""
    
    global MODEL_CONFIG, BALANCE_STRATEGY, OUTPUT_FILE
    
    # Set configuration based on classification type
    if USE_MULTICLASS:
        # Multiclass: MLP with no balancing
        if USE_MLP:
            MODEL_CONFIG = {
                'classifier': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (175,),
                    'activation': 'tanh',
                    'alpha': 0.000636325488567819,
                    'learning_rate': 'constant',
                    'solver': 'sgd',
                    'learning_rate_init': 0.023473801881703782,
                    'max_iter': 7389,
                    'early_stopping': True,
                    'n_iter_no_change': 10,
                    'tol': 0.0001,
                    'random_state': RANDOM_SEED
                }
            }
            BALANCE_STRATEGY = 'tomeklinks'
            OUTPUT_FILE = './results/predictions/predictions_unlabeled_multiclassMLP.csv'
        else:
            MODEL_CONFIG = {
            'classifier': GradientBoostingClassifier,
            'params': {
                'n_estimators': 150,
                'learning_rate': 0.028236885417126437,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 1,
                'subsample': 0.9547690674272925,
                'max_features': 'sqrt',
                'random_state': RANDOM_SEED
            }
        }
        BALANCE_STRATEGY = 'none'
        OUTPUT_FILE = './results/predictions/predictions_unlabeled_MulticlassGB.csv'
    else:

        # Binary: Gradient Boosting with tomeklinks
        MODEL_CONFIG = {
            'classifier': GradientBoostingClassifier,
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.02914494758754147,
                'max_depth': 3,
                'min_samples_split': 7,
                'min_samples_leaf': 4,
                'subsample': 0.8578510331496536,
                'max_features': 'log2',
                'random_state': RANDOM_SEED
            }
        }
        BALANCE_STRATEGY = 'none'
        OUTPUT_FILE = './results/predictions/predictions_unlabeled_binary.csv'
    
    print("="*80)
    print("TRAINING MODEL AND PREDICTING ON UNLABELED DATA")
    print("="*80)
    print()
    
    # Load data
    X_labeled, y_labeled, X_unlabeled, unlabeled_genes, labeled_genes = load_data(USE_MULTICLASS)
    
    # Apply balancing if specified
    if BALANCE_STRATEGY != 'none':
        print(f"\nApplying balancing strategy: {BALANCE_STRATEGY}")
        X_labeled, y_labeled = balance_fold(X_labeled, y_labeled, BALANCE_STRATEGY)
        print(f"After balancing: {X_labeled.shape[0]} samples")
    
    # Create model
    print(f"\nModel configuration:")
    print(f"  Classifier: {MODEL_CONFIG['classifier'].__name__}")
    print(f"  Parameters: {MODEL_CONFIG['params']}")
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MODEL_CONFIG['classifier'](**MODEL_CONFIG['params']))
    ])
    
    # Train model
    print("\nTraining model on all labeled data...")
    pipeline.fit(X_labeled, y_labeled)
    print("Training completed!")
    
    # Get training accuracy
    train_score = pipeline.score(X_labeled, y_labeled)
    print(f"Training accuracy: {train_score:.4f}")
    
    # Predict on unlabeled data
    print(f"\nPredicting on {len(unlabeled_genes)} unlabeled genes...")
    predictions = pipeline.predict(X_unlabeled)
    
    # Get probabilities if available
    try:
        probabilities = pipeline.predict_proba(X_unlabeled)
        has_proba = True
    except:
        has_proba = False
        probabilities = None
    
    # Create results dataframe
    results = pd.DataFrame({
        'gene': unlabeled_genes,
        'predicted_class': predictions
    })
    
    if has_proba:
        if USE_MULTICLASS:
            # For multiclass, add probability for each class
            n_classes = probabilities.shape[1]
            for i in range(n_classes):
                results[f'probability_class_{i}'] = probabilities[:, i]
        else:
            # For binary, add probabilities for both classes
            results['probability_class_0'] = probabilities[:, 0]
            results['probability_class_1'] = probabilities[:, 1]
    
    # Sort by gene name
    results = results.sort_values('gene')
    
    # Create predictions directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save predictions
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPredictions saved to: {OUTPUT_FILE}")
    
    # Print summary
    print(f"\nClass distribution (unlabeled predictions):")
    unique, counts = np.unique(predictions, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {int(cls)}: {count}")
    
    # Show examples of predictions by class
    print(f"\n{'='*80}")
    print("PREDICTION EXAMPLES BY CLASS")
    print(f"{'='*80}")
    
    for cls in sorted(unique):
        cls_int = int(cls)
        class_results = results[results['predicted_class'] == cls]
        
        # Sort by probability (highest confidence first)
        if has_proba:
            if USE_MULTICLASS:
                class_results = class_results.sort_values(f'probability_class_{cls_int}', ascending=False)
            else:
                class_results = class_results.sort_values(f'probability_class_{cls_int}', ascending=False)
        
        n_examples = min(20, len(class_results))
        
        print(f"\nClass {cls_int} - Top {n_examples} predictions (highest confidence):")
        print("-" * 80)
        
        display_df = class_results.head(n_examples).copy()
        
        # Format for display
        if has_proba:
            if USE_MULTICLASS:
                # Show all class probabilities
                prob_cols = [col for col in display_df.columns if col.startswith('probability_class_')]
                display_cols = ['gene'] + prob_cols
            else:
                # Show both class probabilities
                display_cols = ['gene', 'probability_class_0', 'probability_class_1']
        else:
            display_cols = ['gene', 'predicted_class']
        
        print(display_df[display_cols].to_string(index=False))
    
    return results, pipeline


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    results, trained_model = train_and_predict()
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
