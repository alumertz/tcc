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
    average_precision_score,
)
from sklearn.model_selection import cross_validate

try:
    from reports import generate_enhanced_classification_report
except ImportError:
    generate_enhanced_classification_report = None
    print("⚠️ Aviso: 'generate_enhanced_classification_report' não encontrado.")


def detailed_cross_val_score(pipeline, X, y, cv, scoring='average_precision'):
    """
    Executa validação cruzada coletando métricas detalhadas por fold.

    Args:
        pipeline: Estimador ou pipeline scikit-learn
        X (np.ndarray): Features
        y (np.ndarray): Labels
        cv (int ou CV splitter): Número de folds ou estratégia de validação
        scoring (str): Nome da métrica principal

    Returns:
        tuple: (média da métrica principal, lista de métricas detalhadas por fold)
    """
    scoring_metrics = [
        'accuracy', 'precision_weighted', 'recall_weighted',
        'f1_weighted', 'roc_auc', 'average_precision'
    ]

    cv_results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=scoring_metrics,
        return_train_score=True,
        return_estimator=True
    )

    detailed_metrics = []
    n_folds = len(cv_results['test_accuracy'])

    for i in range(n_folds):
        fold = {
            'fold': i + 1,
            **{f'train_{metric}': cv_results[f'train_{metric}'][i] for metric in scoring_metrics},
            **{f'val_{metric}': cv_results[f'test_{metric}'][i] for metric in scoring_metrics},
        }
        detailed_metrics.append(fold)

    main_score = cv_results.get(f'test_{scoring}', [np.nan]).mean()
    return main_score, detailed_metrics


def evaluate_classification_on_test(model, X_test, y_test, return_dict=False):
    """
    Avalia um modelo de classificação no conjunto de teste.

    Args:
        model: Estimador treinado com `.predict()` e `.predict_proba()` ou `.decision_function()`
        X_test (np.ndarray): Dados de teste
        y_test (np.ndarray): Labels verdadeiros
        return_dict (bool): Se True, retorna dicionário com as métricas

    Returns:
        dict ou tuple: Métricas de avaliação ou tupla simples
    """
    y_pred = model.predict(X_test)

    # Probabilidades ou scores
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            raise AttributeError("Modelo não possui predict_proba nem decision_function.")
    except Exception as e:
        print(f"Erro ao obter probabilidades: {e}")
        y_score = np.zeros_like(y_pred)  # fallback seguro

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_score),
        'pr_auc': average_precision_score(y_test, y_score),
    }

    # Relatório customizado, se disponível
    if generate_enhanced_classification_report:
        metrics['classification_report'] = generate_enhanced_classification_report(
            y_test, y_pred, y_score
        )

    if return_dict:
        return metrics

    # Output padrão em texto
    print("\nAvaliação no conjunto de teste (Classificação):")
    for k, v in metrics.items():
        if k != 'classification_report':
            print(f"{k.replace('_', ' ').title()}: {v:.4f}")
    if 'classification_report' in metrics:
        print("\nRelatório detalhado:")
        print(metrics['classification_report'])

    return (
        metrics['accuracy'], metrics['precision'], metrics['recall'],
        metrics['f1'], metrics['roc_auc'], metrics['pr_auc']
    )
