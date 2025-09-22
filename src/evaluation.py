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
from sklearn.model_selection import cross_validate


def detailed_cross_val_score(pipeline, X, y, cv, scoring='average_precision'):
    """
    Executa validação cruzada coletando métricas detalhadas de cada fold
    """
    # Métricas a serem calculadas
    scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc', 'average_precision']
    
    # Executar CV com múltiplas métricas
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, 
        scoring=scoring_metrics,
        return_train_score=True,
        return_estimator=True
    )
    
    # Coletar métricas detalhadas para cada fold
    detailed_metrics = []
    
    for fold_idx in range(len(cv_results['test_accuracy'])):
        fold_metrics = {
            'fold': fold_idx + 1,
            'train_accuracy': cv_results['train_accuracy'][fold_idx],
            'val_accuracy': cv_results['test_accuracy'][fold_idx],
            'train_precision': cv_results['train_precision_weighted'][fold_idx],
            'val_precision': cv_results['test_precision_weighted'][fold_idx],
            'train_recall': cv_results['train_recall_weighted'][fold_idx],
            'val_recall': cv_results['test_recall_weighted'][fold_idx],
            'train_f1': cv_results['train_f1_weighted'][fold_idx],
            'val_f1': cv_results['test_f1_weighted'][fold_idx],
            'train_roc_auc': cv_results['train_roc_auc'][fold_idx],
            'val_roc_auc': cv_results['test_roc_auc'][fold_idx],
            'train_pr_auc': cv_results['train_average_precision'][fold_idx],
            'val_pr_auc': cv_results['test_average_precision'][fold_idx]
        }
        detailed_metrics.append(fold_metrics)
    
    # Retornar média da métrica principal e métricas detalhadas
    main_score = cv_results[f'test_{scoring}'].mean()
    
    return main_score, detailed_metrics


def evaluate_classification_on_test(model, X_test, y_test, return_dict=False):
    """Função para avaliar modelos de classificação no conjunto de teste"""
    from src.reports import generate_enhanced_classification_report
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades para classe positiva
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Gerar relatório customizado
    custom_report = generate_enhanced_classification_report(y_test, y_pred, y_pred_proba)

    if return_dict:
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'classification_report': custom_report
        }
    else:
        print("\n Avaliação no conjunto de teste (Classificação):")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print("\nRelatório detalhado:")
        print(custom_report)
        
        return accuracy, precision, recall, f1, roc_auc, pr_auc
