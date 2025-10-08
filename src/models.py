"""
Módulo para otimização de hiperparâmetros de classificadores de machine learning com Optuna.
Suporta diversos algoritmos da scikit-learn.
"""

import time
import inspect
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Classificadores suportados
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Avaliação e relatórios
from evaluation import detailed_cross_val_score, evaluate_classification_on_test
from reports import save_results_to_file


# =========================
# Funções Auxiliares
# =========================

def build_pipeline(classifier):
    """Cria uma pipeline padrão com normalização + classificador"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])


def _create_classifier(classifier_class, params, seed=30):
    """Inicializa o classificador com ou sem random_state"""
    classifier_signature = inspect.signature(classifier_class).parameters
    if 'random_state' in classifier_signature:
        return classifier_class(random_state=seed, **params)
    return classifier_class(**params)


# =========================
# Função Genérica de Otimização
# =========================

def _optimize_classifier_generic(
    classifier_class,
    param_suggestions_func,
    model_name,
    X, y,
    n_trials=30,
    save_results=True,
    custom_params_processor=None
):
    """Função genérica para otimização de classificadores com validação cruzada."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = param_suggestions_func(trial)
        classifier = _create_classifier(classifier_class, params)
        pipeline = build_pipeline(classifier)

        try:
            start = time.time()
            score, cv_metrics = detailed_cross_val_score(
                pipeline, X_trainval, y_trainval, cv=inner_cv, scoring="average_precision"
            )
            end = time.time()
        except Exception as e:
            print(f"Erro no trial {trial.number}: {e}")
            score = 0.0
            cv_metrics = []
            end = time.time()

        trial_data.append({
            'trial_number': trial.number,
            'params': params.copy(),
            'score': score,
            'time': end - start
        })

        all_cv_metrics.append({
            'trial_number': trial.number,
            'score': score,
            'params': params.copy(),
            'cv_metrics': cv_metrics
        })

        return score

    # Executa a otimização com Optuna
    study = optuna.create_study(direction="maximize")
    total_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_end = time.time()

    # Processa os melhores parâmetros
    best_params = study.best_params
    if custom_params_processor:
        best_params = custom_params_processor(study.best_trial)

    final_classifier = _create_classifier(classifier_class, best_params)
    final_pipeline = build_pipeline(final_classifier)
    final_pipeline.fit(X_trainval, y_trainval)

    test_metrics = evaluate_classification_on_test(final_pipeline, X_test, y_test, return_dict=True)

    # Salvamento opcional
    if save_results:
        results = {
            'trials': trial_data,
            'best_params': best_params,
            'best_score': study.best_value,
            'test_metrics': test_metrics,
            'optimization_time': total_end - total_start,
            'all_cv_metrics': all_cv_metrics
        }
        if model_name == 'decision_tree':
            results['cv_folds_used'] = inner_cv.get_n_splits()

        save_results_to_file(model_name, results)

    # Exibição de resultados
    print(f"\n{model_name.replace('_', ' ').title()}")
    print(f"Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"PR AUC no teste: {test_metrics['pr_auc']:.4f}")

    return final_pipeline


# =========================
# Sugestão de Parâmetros
# =========================

def _suggest_decision_tree_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }

def _suggest_random_forest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }

def _suggest_gradient_boosting_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
    }

def _suggest_hist_gradient_boosting_params(trial):
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
    }

def _suggest_knn_params(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "p": trial.suggest_int("p", 1, 2),
    }

def _suggest_mlp_params(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layers = [trial.suggest_int(f"layer_{i}_size", 10, 200) for i in range(n_layers)]

    return {
        "hidden_layer_sizes": tuple(hidden_layers),
        "activation": trial.suggest_categorical("activation", ["tanh", "relu", "logistic"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "max_iter": trial.suggest_int("max_iter", 200, 1000)
    }

def _process_mlp_params(trial):
    n_layers = trial.params["n_layers"]
    hidden_layers = [trial.params[f"layer_{i}_size"] for i in range(n_layers)]

    return {
        "hidden_layer_sizes": tuple(hidden_layers),
        "activation": trial.params["activation"],
        "alpha": trial.params["alpha"],
        "learning_rate": trial.params["learning_rate"],
        "max_iter": trial.params["max_iter"]
    }

def _suggest_svc_params(trial):
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    params = {
        "kernel": kernel,
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "probability": True
    }

    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    if kernel in ["poly", "rbf", "sigmoid"]:
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])

    return params


# =========================
# Interfaces Públicas (por modelo)
# =========================

def optimize_decision_tree_classifier(X, y, n_trials=30, save_results=True):
    return _optimize_classifier_generic(
        DecisionTreeClassifier, _suggest_decision_tree_params, 'decision_tree', X, y, n_trials, save_results
    )

def optimize_random_forest_classifier(X, y, n_trials=30, save_results=True):
    return _optimize_classifier_generic(
        RandomForestClassifier, _suggest_random_forest_params, 'random_forest', X, y, n_trials, save_results
    )

def optimize_gradient_boosting_classifier(X, y, n_trials=30, save_results=True):
    return _optimize_classifier_generic(
        GradientBoostingClassifier, _suggest_gradient_boosting_params, 'gradient_boosting', X, y, n_trials, save_results
    )

def optimize_hist_gradient_boosting_classifier(X, y, n_trials=30, save_results=True):
    return _optimize_classifier_generic(
        HistGradientBoostingClassifier, _suggest_hist_gradient_boosting_params, 'histogram_gradient_boosting', X, y, n_trials, save_results
    )

def optimize_knn_classifier(X, y, n_trials=30, save_results=True):
    return _optimize_classifier_generic(
        KNeighborsClassifier, _suggest_knn_params, 'k_nearest_neighbors', X, y, n_trials, save_results
    )

def optimize_mlp_classifier(X, y, n_trials=30, save_results=True):
    return _optimize_classifier_generic(
        MLPClassifier, _suggest_mlp_params, 'multi_layer_perceptron', X, y, n_trials, save_results,
        custom_params_processor=_process_mlp_params
    )

def optimize_svc_classifier(X, y, n_trials=30, save_results=True):
    return _optimize_classifier_generic(
        SVC, _suggest_svc_params, 'svc', X, y, n_trials, save_results
    )
