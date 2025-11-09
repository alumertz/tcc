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

# Função unificada para avaliação de modelos holdout e 5-fold cv para parâmetros padrão binay ou multiclass
def evaluate_model_default(model, model_name, X, y, data_source="ana", classification_type="binary"):
    """
    Avalia um modelo com parâmetros padrão usando holdout e 5-fold CV
    Pipeline unificado: StandardScaler + Classifier, métricas binary
    
    Args:
        model: Modelo do scikit-learn com parâmetros padrão
        model_name (str): Nome do modelo
        X (np.array): Features
        y (np.array): Labels
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
        
    Returns:
        dict: Resultados da avaliação
    """
    print(f"\n{'='*80}")
    print(f"AVALIANDO MODELO: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Divisão treino/teste (mesmo random_state do main.py otimizado)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset dividido:")
    print(f"  Treino+Val: {X_trainval.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    
    # Pipeline unificado: SEMPRE com StandardScaler
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])
    
    # Validação cruzada 5-fold no conjunto treino+validação
    print("\nExecutando validação cruzada 5-fold...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
    
    # Métricas para validação cruzada - BINARY
    cv_scores = {
        'accuracy': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='accuracy'),
        'precision': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='precision'),
        'recall': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='recall'),
        'f1': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='f1'),
        'roc_auc': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='roc_auc'),
        'pr_auc': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='average_precision')
    }
    
    # Calcular médias e desvios padrão
    cv_results = {}
    for metric, scores in cv_scores.items():
        cv_results[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores.tolist()
        }
    
    print("Resultados da validação cruzada:")
    for metric, result in cv_results.items():
        print("  " + str(metric).upper() + ": " + "{:.4f}".format(float(result['mean'])) + " ± " + "{:.4f}".format(float(result['std'])))
    
    # Treinar no conjunto treino+validação completo e avaliar no teste
    print("\nTreinando no conjunto completo e avaliando no teste...")
    pipeline.fit(X_trainval, y_trainval)
    
    # Avaliação no conjunto de teste usando função unificada
    test_metrics = evaluate_classification_on_test(pipeline, X_test, y_test, return_dict=True, classification_type=classification_type)
    
    print("Resultados no conjunto de teste:")
    for metric, value in test_metrics.items():
        if metric == 'classification_report':
            continue
        else:
            print("  " + str(metric).upper() + ": " + str(value))
    
    # Obter predições para o relatório de classificação
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Extract probabilities for positive class (binary) or keep all classes (multiclass)
    if classification_type == 'binary' and y_pred_proba.shape[1] == 2:
        y_pred_proba_positive = y_pred_proba[:, 1]
    else:
        y_pred_proba_positive = y_pred_proba
    
    # Relatório de classificação detalhado
    from src.reports import generate_enhanced_classification_report
    class_report = generate_enhanced_classification_report(y_test, y_pred, y_pred_proba_positive)
    print("\nRelatório de classificação:\n" + str(class_report))
    
    # Salvar resultados usando função unificada
    # Convert parameters to JSON-serializable format
    clean_params = {}
    for key, value in pipeline.get_params().items():
        clean_params[key] = str(value)
    
    results_data = {
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'classification_report': class_report,
        'parameters': clean_params,
        # Save predictions for plotting
        'test_predictions': {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba_positive.tolist() if hasattr(y_pred_proba_positive, 'tolist') else y_pred_proba_positive,
            'test_indices': X_test.index.tolist() if hasattr(X_test, 'index') else list(range(len(X_test)))
        }
    }
    
    save_model_results_unified(model_name, results_data, mode="default", 
                             data_source=data_source, classification_type=classification_type)
    
    results = {
        'model_name': model_name,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'classification_report': class_report,
        'model': pipeline,
        'status': 'success'
    }
    
    return results

# Logica de avaliação unificada para classificação binária e multiclasse
def evaluate_classification_on_test(model, X_test, y_test, return_dict=False, classification_type='binary'):
    """Função para avaliar modelos de classificação no conjunto de teste

    Args:
        model: estimador treinado (deve implementar predict and predict_proba)
        X_test, y_test: dados de teste
        return_dict: se True retorna dicionário com métricas
        classification_type: 'binary' ou 'multiclass' (afeta como ROC/PR e médias são calculadas)
    """
    from reports import generate_enhanced_classification_report
    from sklearn.preprocessing import label_binarize

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    if classification_type == 'multiclass':
        # Use weighted averages for multiclass
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # ROC AUC (multiclass OVR) and 
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

        # PR AUC per class averaged (macro)
        classes = np.unique(y_test)
        y_test_b = label_binarize(y_test, classes=classes)
        pr_auc = average_precision_score(y_test_b, y_proba, average='weighted')

        # Generate a multiclass-friendly text report
        custom_report = generate_enhanced_classification_report(y_test, y_pred, y_proba)

    else:
        # Binary classification (default behavior)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        # Probabilities for positive class
        if y_proba.ndim == 2 and y_proba.shape[1] > 1:
            y_pred_proba = y_proba[:, 1]
        else:
            # If model returns a single column or already shaped vector
            y_pred_proba = y_proba.ravel()

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        custom_report = generate_enhanced_classification_report(y_test, y_pred, y_pred_proba)

    if return_dict:
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else float('nan'),
            'pr_auc': float(pr_auc) if not np.isnan(pr_auc) else float('nan'),
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
