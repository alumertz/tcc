"""
Módulo para avaliação de modelos de machine learning.
Contém funções para validação cruzada detalhada e avaliação em conjunto de teste.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.reports import save_model_results_unified
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')


from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import os

def evaluate_model_default(model, model_name, X, y, experiment_dir, data_source="ana", classification_type="binary"):
    """
    Avalia modelo com 5-fold CV (80/20 split), salva métricas de treino e teste por fold.
    """
    # Split 80/20
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])
        pipeline.fit(X_train, y_train)

        # Train metrics
        y_pred_train = pipeline.predict(X_train)
        y_proba_train = pipeline.predict_proba(X_train)
        train_metrics = get_metrics(y_train, y_pred_train, y_proba_train, classification_type)

        # Validation metrics
        y_pred_val = pipeline.predict(X_val)
        y_proba_val = pipeline.predict_proba(X_val)
        val_metrics = get_metrics(y_val, y_pred_val, y_proba_val, classification_type)

        folds_metrics.append({
            'fold': fold,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })

    # Final model on all trainval, test metrics
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])
    pipeline.fit(X_trainval, y_trainval)
    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)
    test_metrics = get_metrics(y_test, y_pred_test, y_proba_test, classification_type)

    # Chama função de relatório
    from src.reports import save_detailed_results_txt_by_fold
    save_detailed_results_txt_by_fold(
        model_name=model_name,
        all_folds_trials=[{
            'fold': m['fold'],
            'train_metrics': m['train_metrics'],
            'test_metrics': m['val_metrics'],
            'trials': [],  # Não há trials no default
            'best_params': {}  # Adiciona chave vazia para compatibilidade
        } for m in folds_metrics],
        output_path=f"{experiment_dir}/{model_name.lower().replace(' ', '_')}/default_results.txt",
        final_best_params={k: str(v) for k, v in pipeline.get_params().items()}
    )

    return {
        'model_name': model_name,
        'folds_metrics': folds_metrics,
        'test_metrics': test_metrics,
        'model': pipeline
    }

def get_metrics(y_true, y_pred, y_proba, classification_type):
    average = 'weighted' if classification_type == 'multiclass' else 'binary'
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average),
        'roc_auc': roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted') if classification_type == 'multiclass' else roc_auc_score(y_true, y_proba[:, 1]),
        'pr_auc': average_precision_score(y_true, y_proba, average='weighted') if classification_type == 'multiclass' else average_precision_score(y_true, y_proba[:, 1])
    }
    return metrics