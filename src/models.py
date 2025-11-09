"""
Módulo para otimização de hiperparâmetros de modelos de machine learning.
Contém funções de otimização usando Optuna para diferentes algoritmos de classificação.
Implementa validação cruzada aninhada (nested cross-validation) para avaliação não-enviesada.
"""

import time
import numpy as np
import optuna
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
    
def _optimize_classifier_generic(classifier_class, param_suggestions_func, model_name, X, y, 
                               n_trials=100, save_results=True, custom_params_processor=None, 
                               return_test_metrics=False, fixed_params=None, data_source="ana",
                               classification_type="binary", use_nested_cv=True, outer_cv_folds=5):
    """
    Função genérica para otimização de hiperparâmetros de classificadores com nested cross-validation.
    Implementa nested cross-validation para avaliação imparcial do modelo.
    
    O loop externo divide os dados em treino/teste para avaliação final.
    O loop interno otimiza hiperparâmetros usando cross-validation no conjunto de treino.
    """
    
    
    # Configurar cross-validation estratificado
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    
    nested_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'best_params_per_fold': []
    }
    
    print(f"\nIniciando Nested Cross-Validation para {model_name}...")
    print(f"Configuração: {outer_cv_folds} folds externos, {n_trials} trials por fold")
    
    fold_number = 1
    for train_idx, test_idx in outer_cv.split(X, y):
        print(f"\nFold {fold_number}/{outer_cv_folds}")
        
        # Dividir dados para este fold - funciona com numpy arrays e pandas DataFrames
        if hasattr(X, 'iloc'):
            # pandas DataFrame
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        else:
            # numpy array
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            
        if hasattr(y, 'iloc'):
            # pandas Series
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        else:
            # numpy array
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Otimização no conjunto de treino (loop interno)
        def objective(trial):
            # Obter parâmetros sugeridos
            params = param_suggestions_func(trial)
            
            # Aplicar processamento customizado se fornecido
            if custom_params_processor:
                params = custom_params_processor(params)
                
            # Combinar com parâmetros fixos
            if fixed_params:
                params.update(fixed_params)
            
            # Criar pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', classifier_class(**params))
            ])
            
            # Cross-validation interno (5-fold)
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X_train_fold, y_train_fold, 
                                   cv=inner_cv, scoring='average_precision')
            
            return scores.mean()
        
        # Criar estudo Optuna para este fold
        study_name = f"{model_name}_fold_{fold_number}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=None,  # Em memória
            load_if_exists=False
        )
        
        # Otimizar
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Treinar modelo final com melhores parâmetros neste fold
        best_params = study.best_params.copy()
        if custom_params_processor:
            best_params = custom_params_processor(best_params)
        if fixed_params:
            best_params.update(fixed_params)
            
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier_class(**best_params))
        ])
        
        # Treinar no conjunto de treino do fold
        final_pipeline.fit(X_train_fold, y_train_fold)
        
        # Avaliar no conjunto de teste do fold
        y_pred_fold = final_pipeline.predict(X_test_fold)
        # Try to get probabilities safely
        try:
            y_pred_proba_fold = final_pipeline.predict_proba(X_test_fold)
        except AttributeError:
            # If pipeline doesn't have predict_proba, try to get from the classifier
            classifier = final_pipeline.named_steps['classifier']
            if hasattr(classifier, 'predict_proba'):
                # Apply scaler transform before prediction if needed
                X_test_scaled = final_pipeline.named_steps['scaler'].transform(X_test_fold)
                y_pred_proba_fold = classifier.predict_proba(X_test_scaled)
            else:
                raise AttributeError("Neither Pipeline nor classifier implement predict_proba")
        
        # Calcular métricas para este fold
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
        precision_fold, recall_fold, f1_fold, _ = precision_recall_fscore_support(
            y_test_fold, y_pred_fold, average='weighted' if classification_type == "multiclass" else 'binary'
        )
        
        if classification_type == "multiclass":
            roc_auc_fold = roc_auc_score(y_test_fold, y_pred_proba_fold, multi_class='ovr', average='weighted')
        else:
            roc_auc_fold = roc_auc_score(y_test_fold, y_pred_proba_fold[:, 1])
        
        # Armazenar resultados deste fold
        nested_scores['accuracy'].append(accuracy_fold)
        nested_scores['precision'].append(precision_fold)
        nested_scores['recall'].append(recall_fold)
        nested_scores['f1'].append(f1_fold)
        nested_scores['roc_auc'].append(roc_auc_fold)
        nested_scores['best_params_per_fold'].append(best_params)
        
        print(f"Fold {fold_number} - F1: {f1_fold:.4f}, Accuracy: {accuracy_fold:.4f}, ROC-AUC: {roc_auc_fold:.4f}")
        fold_number += 1
    
    # Calcular estatísticas agregadas
    aggregated_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = nested_scores[metric]
        aggregated_metrics[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    
    print(f"\n=== Resultados Nested Cross-Validation - {model_name} ===")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        mean_score = aggregated_metrics[metric]['mean']
        std_score = aggregated_metrics[metric]['std']
        print(f"{metric.upper()}: {mean_score:.4f} (±{std_score:.4f})")
    
    # Treinar modelo final em todos os dados para retorno
    # Usar os melhores parâmetros do melhor fold (baseado em F1)
    best_fold_idx = np.argmax(nested_scores['f1'])
    final_best_params = nested_scores['best_params_per_fold'][best_fold_idx]
    
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier_class(**final_best_params))
    ])
    final_pipeline.fit(X, y)
    
    # Salvar resultados se solicitado
    if save_results:
        from src.reports import save_nested_cv_results
        save_nested_cv_results(
            model_name=model_name,
            aggregated_metrics=aggregated_metrics,
            best_params_per_fold=nested_scores['best_params_per_fold'],
            data_source=data_source,
            classification_type=classification_type,
            n_trials=n_trials,
            outer_cv_folds=outer_cv_folds
        )
    
    if return_test_metrics:
        return final_pipeline, aggregated_metrics
    else:
        return final_pipeline


# Funções de sugestão de parâmetros para cada modelo
def _suggest_decision_tree_params(trial):
    """Sugestões de parâmetros para Decision Tree"""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def _suggest_random_forest_params(trial):
    """Sugestões de parâmetros para Random Forest"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
    }


def _suggest_gradient_boosting_params(trial):
    """Sugestões de parâmetros para Gradient Boosting"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0)
    }


def _suggest_hist_gradient_boosting_params(trial):
    """Sugestões de parâmetros para Histogram Gradient Boosting"""
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0)
    }


def _suggest_knn_params(trial):
    """Sugestões de parâmetros para KNN"""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "p": trial.suggest_int("p", 1, 2)  # 1 for manhattan, 2 for euclidean
    }


def _suggest_mlp_params(trial):
    """Sugestões de parâmetros para MLP"""
    # Sugerir arquitetura da rede
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layer_sizes = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 10, 200)
        hidden_layer_sizes.append(layer_size)
    
    return {
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "activation": trial.suggest_categorical("activation", ["tanh", "relu", "logistic"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "max_iter": trial.suggest_int("max_iter", 200, 1000)
    }


def _suggest_svc_params(trial):
    """Sugestões de parâmetros para SVC otimizadas para evitar execução infinita"""
    # Priorizar kernels mais eficientes e limitar opções problemáticas
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    
    params = {
        "kernel": kernel,
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),  # Range mais restrito
        "probability": True,  # Necessário para predict_proba
        "max_iter": 1000,  # Limitar iterações para evitar execução infinita
        "tol": 1e-3,  # Tolerância menos rigorosa para convergência mais rápida
        "cache_size": 200,  # Aumentar cache para melhor performance
    }
    
    # Adicionar parâmetros específicos do kernel com restrições
    if kernel == "rbf":
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    return params


def _suggest_catboost_params(trial):
    """Sugestões de parâmetros para CatBoost"""
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "verbose": False,  # Silenciar logs durante otimização
        "allow_writing_files": False  # Não escrever arquivos de log
    }


def _process_mlp_params(best_params):
    """Processa parâmetros do MLP para o modelo final.

    Espera um dicionário de parâmetros (ex.: param_suggestions_func(trial) ou study.best_params).
    Retorna dicionário pronto para passar ao MLPClassifier.
    """
    if not isinstance(best_params, dict):
        raise ValueError("_process_mlp_params espera um dict com parâmetros do MLP")

    # Prefer explicit hidden_layer_sizes if present, senão montar a partir de n_layers / layer_{i}_size
    if "hidden_layer_sizes" in best_params and best_params.get("hidden_layer_sizes") is not None:
        hidden = tuple(best_params["hidden_layer_sizes"])
    else:
        n_layers = int(best_params.get("n_layers", 0) or 0)
        hidden_list = []
        for i in range(n_layers):
            key = f"layer_{i}_size"
            if key in best_params:
                hidden_list.append(int(best_params[key]))
        hidden = tuple(hidden_list)

    return {
        "hidden_layer_sizes": hidden,
        "activation": best_params.get("activation", "relu"),
        "alpha": float(best_params.get("alpha", 1e-4)),
        "learning_rate": best_params.get("learning_rate", "constant"),
        "max_iter": int(best_params.get("max_iter", 200)),
        "early_stopping": bool(best_params.get("early_stopping", True))
    }


def optimize_decision_tree_classifier(X, y, n_trials=30, save_results=True, fixed_params=None, 
                                    data_source="ana", classification_type="binary", 
                                    use_nested_cv=True, outer_cv_folds=5):
    """Otimização de hiperparâmetros para Decision Tree Classifier usando Optuna"""
    return _optimize_classifier_generic(
        DecisionTreeClassifier,
        _suggest_decision_tree_params,
        'decision_tree',
        X, y, n_trials, save_results, 
        custom_params_processor=None,
        return_test_metrics=True,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        use_nested_cv=use_nested_cv, outer_cv_folds=outer_cv_folds
    )


def optimize_random_forest_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                                     data_source="ana", classification_type="binary", 
                                     use_nested_cv=True, outer_cv_folds=5):
    """Otimização de hiperparâmetros para Random Forest Classifier usando Optuna"""
    return _optimize_classifier_generic(
        RandomForestClassifier,
        _suggest_random_forest_params,
        'random_forest',
        X, y, n_trials, save_results, 
        custom_params_processor=None,
        return_test_metrics=True,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        use_nested_cv=use_nested_cv, outer_cv_folds=outer_cv_folds
    )


def optimize_gradient_boosting_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                                        data_source="ana", classification_type="binary", 
                                        use_nested_cv=True, outer_cv_folds=5):
    """Otimização de hiperparâmetros para Gradient Boosting Classifier usando Optuna"""
    return _optimize_classifier_generic(
        GradientBoostingClassifier,
        _suggest_gradient_boosting_params,
        'gradient_boosting',
        X, y, n_trials, save_results, 
        custom_params_processor=None,
        return_test_metrics=True,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        use_nested_cv=use_nested_cv, outer_cv_folds=outer_cv_folds
    )


def optimize_hist_gradient_boosting_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                                             data_source="ana", classification_type="binary", 
                                             use_nested_cv=True, outer_cv_folds=5):
    """Otimização de hiperparâmetros para Histogram Gradient Boosting Classifier usando Optuna"""
    return _optimize_classifier_generic(
        HistGradientBoostingClassifier,
        _suggest_hist_gradient_boosting_params,
        'histogram_gradient_boosting',
        X, y, n_trials, save_results, 
        custom_params_processor=None,
        return_test_metrics=True,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        use_nested_cv=use_nested_cv, outer_cv_folds=outer_cv_folds
    )


def optimize_knn_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                           data_source="ana", classification_type="binary", 
                           use_nested_cv=True, outer_cv_folds=5):
    """Otimização de hiperparâmetros para KNN Classifier usando Optuna"""
    return _optimize_classifier_generic(
        KNeighborsClassifier,
        _suggest_knn_params,
        'k_nearest_neighbors',
        X, y, n_trials, save_results, 
        custom_params_processor=None,
        return_test_metrics=True,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        use_nested_cv=use_nested_cv, outer_cv_folds=outer_cv_folds
    )


def optimize_mlp_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                          data_source="ana", classification_type="binary", 
                          use_nested_cv=True, outer_cv_folds=5):
    """Otimização de hiperparâmetros para MLP Classifier usando Optuna"""
    return _optimize_classifier_generic(
        MLPClassifier,
        _suggest_mlp_params,
        'multi_layer_perceptron',
        X, y, n_trials, save_results,
        custom_params_processor=_process_mlp_params,
        return_test_metrics=True,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        use_nested_cv=use_nested_cv, outer_cv_folds=outer_cv_folds
    )


def optimize_svc_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                          data_source="ana", classification_type="binary", 
                          use_nested_cv=True, outer_cv_folds=5):
    """Otimização de hiperparâmetros para SVC usando Optuna"""
    # Ensure probability=True is always set
    enforced_params = {"probability": True}
    if fixed_params:
        enforced_params.update(fixed_params)
    
    return _optimize_classifier_generic(
        SVC,
        _suggest_svc_params,
        'svc',
        X, y, n_trials, save_results, 
        custom_params_processor=None,
        return_test_metrics=True,
        fixed_params=enforced_params,
        data_source=data_source, classification_type=classification_type,
        use_nested_cv=use_nested_cv, outer_cv_folds=outer_cv_folds
    )


def optimize_catboost_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                               data_source="ana", classification_type="binary", 
                               use_nested_cv=True, outer_cv_folds=5):
    """Otimização de hiperparâmetros para CatBoost Classifier usando Optuna"""
    return _optimize_classifier_generic(
        CatBoostClassifier,
        _suggest_catboost_params,
        'catboost',
        X, y, n_trials, save_results, 
        custom_params_processor=None,
        return_test_metrics=True,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        use_nested_cv=use_nested_cv, outer_cv_folds=outer_cv_folds
    )
