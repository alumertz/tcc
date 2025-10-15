"""
Módulo para otimização de hiperparâmetros de classificadores de machine learning com Optuna.
Suporta diversos algoritmos da scikit-learn.
"""

import time
import inspect
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
from catboost import CatBoostClassifier

# Avaliação e relatórios
from evaluation import detailed_cross_val_score, evaluate_classification_on_test
from reports import save_simple_results_to_file

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
# Função Genérica
# =========================
def run_classifier_with_params(
    classifier_class,
    model_name,
    X, y,
    params: dict,
    save_results=True,
    omics_used=None
):
    """Executa treinamento + validação cruzada + teste com os parâmetros fornecidos."""

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifier = _create_classifier(classifier_class, params)
    pipeline = build_pipeline(classifier)

    # Validação cruzada
    try:
        start = time.time()
        cv_score, cv_metrics = detailed_cross_val_score(
            pipeline, X_trainval, y_trainval, cv=inner_cv, scoring="average_precision"
        )
        end = time.time()
    except Exception as e:
        print(f"Erro durante CV: {e}")
        return None

    # Treinamento final e avaliação
    pipeline.fit(X_trainval, y_trainval)
    test_metrics = evaluate_classification_on_test(pipeline, X_test, y_test, return_dict=True)

    # Salvamento opcional
    if save_results:
        results = {
            'params': params,
            'cv_score': cv_score,
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'training_time': end - start,
        }
        if omics_used:
            results['omics_used'] = omics_used

        save_simple_results_to_file(model_name, results)

    # Exibição
    print(f"\n{model_name.replace('_', ' ').title()}")
    print(f"PR AUC (CV): {cv_score:.4f}")
    print(f"Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"PR AUC no teste: {test_metrics['pr_auc']:.4f}")

    return pipeline

# =========================
# Interfaces Públicas (por modelo)
# =========================

def train_decision_tree_classifier(X, y, params, save_results=True, omics_used=None):
    return run_classifier_with_params(
        DecisionTreeClassifier, 'decision_tree', X, y, params, save_results=save_results, omics_used=omics_used
    )

def train_random_forest_classifier(X, y, params, save_results=True, omics_used=None):
    return run_classifier_with_params(
        RandomForestClassifier, 'random_forest', X, y, params, save_results, omics_used=omics_used
    )

def train_gradient_boosting_classifier(X, y, params, save_results=True, omics_used=None):
    return run_classifier_with_params(
        GradientBoostingClassifier, 'gradient_boosting', X, y, params, save_results, omics_used=omics_used
    )

def train_hist_gradient_boosting_classifier(X, y, params, save_results=True, omics_used=None):
    return run_classifier_with_params(
        HistGradientBoostingClassifier, 'histogram_gradient_boosting', X, y, params, save_results, omics_used=omics_used
    )

def train_knn_classifier(X, y, params, save_results=True, omics_used=None):
    return run_classifier_with_params(
        KNeighborsClassifier, 'k_nearest_neighbors', X, y, params, save_results, omics_used=omics_used
    )

def train_mlp_classifier(X, y, params, save_results=True, omics_used=None):
    return run_classifier_with_params(
        MLPClassifier, 'multi_layer_perceptron', X, y, params, save_results, omics_used=omics_used
    )

def train_svc_classifier(X, y, params, save_results=True, omics_used=None):
    return run_classifier_with_params(
        SVC, 'support_vector_classifier', X, y, params, save_results, omics_used=omics_used
    )


def train_catboost_classifier(X, y, params, save_results=True, omics_used=None):
    return run_classifier_with_params(
        CatBoostClassifier, 'catboost', X, y, params, save_results, omics_used=omics_used
    )
