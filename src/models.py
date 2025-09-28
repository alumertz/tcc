"""
Módulo para otimização de hiperparâmetros de modelos de machine learning.
Contém funções de otimização usando Optuna para diferentes algoritmos de classificação.
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from src.evaluation import detailed_cross_val_score, evaluate_classification_on_test
from src.reports import save_results_to_file


def _optimize_classifier_generic(
    classifier_class, 
    param_suggestions_func, 
    model_name, 
    X, y, 
    n_trials=30, 
    save_results=True,
    custom_params_processor=None
):
    """
    Função genérica para otimização de hiperparâmetros de classificadores.
    
    Parameters:
    -----------
    classifier_class : sklearn classifier class
        Classe do classificador a ser otimizado
    param_suggestions_func : function
        Função que recebe um trial do Optuna e retorna os parâmetros sugeridos
    model_name : str
        Nome do modelo para salvamento e logs
    X, y : array-like
        Features e target
    n_trials : int
        Número de trials para otimização
    save_results : bool
        Se deve salvar os resultados
    custom_params_processor : function, optional
        Função para processar parâmetros antes de criar o modelo final
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Pipeline otimizado
    """
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []

    # Validação cruzada estratificada interna (5-fold)
    n_splits = 5
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial):
        params = param_suggestions_func(trial)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier_class(random_state=30, **params))
        ])

        start = time.time()
        try:
            # Usar validação cruzada detalhada
            score, cv_metrics = detailed_cross_val_score(
                pipeline, X_trainval, y_trainval,
                cv=inner_cv,
                scoring="average_precision"
            )
            
            # Salvar métricas de CV de todos os trials
            all_cv_metrics.append({
                'trial_number': trial.number,
                'score': score,
                'params': params.copy(),
                'cv_metrics': cv_metrics
            })
                
        except Exception as e:
            print(f"Erro no trial {trial.number}: {e}")
            score = 0.0
            cv_metrics = []
            all_cv_metrics.append({
                'trial_number': trial.number,
                'score': score,
                'params': params.copy(),
                'cv_metrics': cv_metrics
            })
            
        end = time.time()

        trial_info = {
            'trial_number': trial.number,
            'params': params,
            'score': score,
            'time': end - start
        }
        trial_data.append(trial_info)

        return score

    # Otimização com Optuna
    study = optuna.create_study(direction="maximize")
    total_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_end = time.time()

    # Processar parâmetros se necessário (para casos especiais como MLP)
    final_params = study.best_params.copy()
    if custom_params_processor:
        final_params = custom_params_processor(study.best_trial)
    
    # Para SVC, garantir que probability=True esteja sempre presente
    if classifier_class == SVC:
        final_params["probability"] = True

    # Treina modelo final com best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier_class(random_state=30, **final_params))
    ])
    final_pipeline.fit(X_trainval, y_trainval)

    # Avalia no conjunto de teste
    test_metrics = evaluate_classification_on_test(final_pipeline, X_test, y_test, return_dict=True)

    # Salva resultados se solicitado
    if save_results:
        results = {
            'trials': trial_data,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'test_metrics': test_metrics,
            'optimization_time': total_end - total_start,
            'all_cv_metrics': all_cv_metrics
        }
        # Adicionar cv_folds_used para Decision Tree (compatibilidade)
        if model_name == 'decision_tree':
            results['cv_folds_used'] = n_splits
            
        save_results_to_file(model_name, results)
    
    print(f"{model_name.replace('_', ' ').title()} - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"{model_name.replace('_', ' ').title()} - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"{model_name.replace('_', ' ').title()} - PR AUC no teste: {test_metrics['pr_auc']:.4f}")
    if model_name == 'decision_tree':
        print(f"{model_name.replace('_', ' ').title()} - CV folds utilizados: {n_splits}")

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


def _process_mlp_params(best_trial):
    """Processa parâmetros do MLP para o modelo final"""
    n_layers = best_trial.params["n_layers"]
    hidden_layer_sizes = []
    for i in range(n_layers):
        layer_size = best_trial.params[f"layer_{i}_size"]
        hidden_layer_sizes.append(layer_size)
    
    return {
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "activation": best_trial.params["activation"],
        "alpha": best_trial.params["alpha"],
        "learning_rate": best_trial.params["learning_rate"],
        "max_iter": best_trial.params["max_iter"]
    }


def optimize_decision_tree_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Decision Tree Classifier usando Optuna"""
    return _optimize_classifier_generic(
        DecisionTreeClassifier,
        _suggest_decision_tree_params,
        'decision_tree',
        X, y, n_trials, save_results
    )


def optimize_random_forest_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Random Forest Classifier usando Optuna"""
    return _optimize_classifier_generic(
        RandomForestClassifier,
        _suggest_random_forest_params,
        'random_forest',
        X, y, n_trials, save_results
    )


def optimize_gradient_boosting_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Gradient Boosting Classifier usando Optuna"""
    return _optimize_classifier_generic(
        GradientBoostingClassifier,
        _suggest_gradient_boosting_params,
        'gradient_boosting',
        X, y, n_trials, save_results
    )


def optimize_hist_gradient_boosting_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Histogram Gradient Boosting Classifier usando Optuna"""
    return _optimize_classifier_generic(
        HistGradientBoostingClassifier,
        _suggest_hist_gradient_boosting_params,
        'histogram_gradient_boosting',
        X, y, n_trials, save_results
    )


def optimize_knn_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para KNN Classifier usando Optuna"""
    return _optimize_classifier_generic(
        KNeighborsClassifier,
        _suggest_knn_params,
        'k_nearest_neighbors',
        X, y, n_trials, save_results
    )


def optimize_mlp_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para MLP Classifier usando Optuna"""
    return _optimize_classifier_generic(
        MLPClassifier,
        _suggest_mlp_params,
        'multi_layer_perceptron',
        X, y, n_trials, save_results,
        custom_params_processor=_process_mlp_params
    )


def optimize_svc_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para SVC usando Optuna"""
    return _optimize_classifier_generic(
        SVC,
        _suggest_svc_params,
        'svc',
        X, y, n_trials, save_results
    )


def optimize_catboost_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para CatBoost Classifier usando Optuna"""
    return _optimize_classifier_generic(
        CatBoostClassifier,
        _suggest_catboost_params,
        'catboost',
        X, y, n_trials, save_results
    )
