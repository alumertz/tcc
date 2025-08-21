import time
import os
import json
from datetime import datetime

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
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score, 
    train_test_split, 
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def get_optimal_cv_folds(y):
    """
    Determina o número otimal de folds baseado no tamanho da classe minoritária
    
    Args:
        y (array): Labels do dataset
        
    Returns:
        int: Número de folds recomendado
    """
    unique, counts = np.unique(y, return_counts=True)
    min_class_size = min(counts)
    
    if min_class_size < 6:
        return 3  # Para datasets muito pequenos
    elif min_class_size < 10:
        return 3  # Para datasets pequenos
    else:
        return 5  # Para datasets maiores


def save_results_to_file(model_name, results, results_dir="/Users/i583975/git/tcc/artigo/results"):
    """Salva os resultados em arquivos organizados por modelo"""
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salva as trials do Optuna
    trials_file = os.path.join(model_dir, f"trials_{timestamp}.json")
    with open(trials_file, 'w') as f:
        json.dump(results['trials'], f, indent=2, default=str)
    
    # Salva os resultados do teste
    test_results_file = os.path.join(model_dir, f"test_results_{timestamp}.txt")
    with open(test_results_file, 'w') as f:
        f.write(f"RESULTADOS DO MODELO: {model_name.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("AVALIAÇÃO NO CONJUNTO DE TESTE (CLASSIFICAÇÃO):\n")
        f.write("-"*50 + "\n")
        f.write(f"Acurácia: {results['test_metrics']['accuracy']:.4f}\n")
        f.write(f"Precisão: {results['test_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {results['test_metrics']['recall']:.4f}\n")
        f.write(f"F1-Score: {results['test_metrics']['f1']:.4f}\n")
        f.write(f"ROC AUC: {results['test_metrics']['roc_auc']:.4f}\n")
        f.write(f"PR AUC: {results['test_metrics']['pr_auc']:.4f}\n\n")
        
        f.write("RELATÓRIO DETALHADO:\n")
        f.write("-"*30 + "\n")
        f.write(results['test_metrics']['classification_report'])
        f.write("\n\n")
        
        f.write("MELHORES HIPERPARÂMETROS:\n")
        f.write("-"*30 + "\n")
        f.write(json.dumps(results['best_params'], indent=2))
        f.write("\n\n")
        
        f.write("HISTÓRICO DE OTIMIZAÇÃO:\n")
        f.write("-"*30 + "\n")
        f.write(f"Número de trials: {len(results['trials'])}\n")
        f.write(f"Melhor score (CV): {results['best_score']:.4f}\n")
        f.write(f"Tempo total de otimização: {results['optimization_time']:.2f} segundos\n")
    
    print(f"Resultados salvos em: {model_dir}")
    return trials_file, test_results_file


def evaluate_classification_on_test(model, X_test, y_test, return_dict=False):
    """Função para avaliar modelos de classificação no conjunto de teste"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades para classe positiva
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    if return_dict:
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'classification_report': report
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
        print(classification_report(y_test, y_pred))
        
        return accuracy, precision, recall, f1, roc_auc, pr_auc


def optimize_decision_tree_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Decision Tree Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []

    # Validação cruzada estratificada interna com folds adaptativos
    n_splits = get_optimal_cv_folds(y_trainval)
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(random_state=30, **params))
        ])

        start = time.time()
        try:
            score = cross_val_score(
                pipeline, X_trainval, y_trainval,
                cv=inner_cv,
                scoring="accuracy",
                error_score='raise'
            ).mean()
        except ValueError as e:
            print(f"Erro no trial {trial.number}: {e}")
            score = 0.0
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

    # Treina modelo final com best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(random_state=30, **study.best_params))
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
            'cv_folds_used': n_splits
        }
        save_results_to_file('decision_tree', results)
    
    print(f"Decision Tree - Melhor acurácia (CV): {study.best_value:.4f}")
    print(f"Decision Tree - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"Decision Tree - CV folds utilizados: {n_splits}")

    return final_pipeline


def optimize_random_forest_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Random Forest Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []

    # Validação cruzada estratificada interna (5-fold)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=30, **params))
        ])

        start = time.time()
        score = cross_val_score(
            pipeline, X_trainval, y_trainval,
            cv=inner_cv,
            scoring="accuracy"
        ).mean()
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

    # Treina modelo final com best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=30, **study.best_params))
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
            'optimization_time': total_end - total_start
        }
        save_results_to_file('random_forest', results)
    
    print(f"Random Forest - Melhor acurácia (CV): {study.best_value:.4f}")
    print(f"Random Forest - Acurácia no teste: {test_metrics['accuracy']:.4f}")

    return final_pipeline


def optimize_gradient_boosting_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Gradient Boosting Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(X[:20])
    print(y[:20])
    print(np.unique(y, return_counts=True))

    trial_data = []

    smote = SMOTE(random_state=42, k_neighbors=3)

    X_trainval, y_trainval = smote.fit_resample(X_trainval, y_trainval)

    # Validação cruzada estratificada interna (5-fold)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            #"subsample": trial.suggest_float("subsample", 0.8, 1.0)
            "subsample": 1.0
        }


        pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(random_state=30, **params))
        ])
        
        start = time.time()
        score = cross_val_score(
            pipeline, X_trainval, y_trainval,
            cv=inner_cv,
            scoring="accuracy"
        ).mean()
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

    # Treina modelo final com best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(random_state=30, **study.best_params))
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
            'optimization_time': total_end - total_start
        }
        save_results_to_file('gradient_boosting', results)
    
    print(f"Gradient Boosting - Melhor acurácia (CV): {study.best_value:.4f}")
    print(f"Gradient Boosting - Acurácia no teste: {test_metrics['accuracy']:.4f}")

    return final_pipeline


def optimize_hist_gradient_boosting_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Histogram Gradient Boosting Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []

    # Validação cruzada estratificada interna (5-fold)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0)
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", HistGradientBoostingClassifier(random_state=30, **params))
        ])

        start = time.time()
        score = cross_val_score(
            pipeline, X_trainval, y_trainval,
            cv=inner_cv,
            scoring="accuracy"
        ).mean()
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

    # Treina modelo final com best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", HistGradientBoostingClassifier(random_state=30, **study.best_params))
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
            'optimization_time': total_end - total_start
        }
        save_results_to_file('hist_gradient_boosting', results)
    
    print(f"Hist Gradient Boosting - Melhor acurácia (CV): {study.best_value:.4f}")
    print(f"Hist Gradient Boosting - Acurácia no teste: {test_metrics['accuracy']:.4f}")

    return final_pipeline


def optimize_knn_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para KNN Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []

    # Validação cruzada estratificada interna (5-fold)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            "p": trial.suggest_int("p", 1, 2)  # 1 for manhattan, 2 for euclidean
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(**params))
        ])

        start = time.time()
        score = cross_val_score(
            pipeline, X_trainval, y_trainval,
            cv=inner_cv,
            scoring="accuracy"
        ).mean()
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

    # Treina modelo final com best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(**study.best_params))
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
            'optimization_time': total_end - total_start
        }
        save_results_to_file('knn', results)
    
    print(f"KNN - Melhor acurácia (CV): {study.best_value:.4f}")
    print(f"KNN - Acurácia no teste: {test_metrics['accuracy']:.4f}")

    return final_pipeline


def optimize_mlp_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para MLP Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []

    # Validação cruzada estratificada interna (5-fold)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        # Sugerir arquitetura da rede
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layer_sizes = []
        for i in range(n_layers):
            layer_size = trial.suggest_int(f"layer_{i}_size", 10, 200)
            hidden_layer_sizes.append(layer_size)
        
        params = {
            "hidden_layer_sizes": tuple(hidden_layer_sizes),
            "activation": trial.suggest_categorical("activation", ["tanh", "relu", "logistic"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
            "max_iter": trial.suggest_int("max_iter", 200, 1000)
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(random_state=30, **params))
        ])

        start = time.time()
        score = cross_val_score(
            pipeline, X_trainval, y_trainval,
            cv=inner_cv,
            scoring="accuracy"
        ).mean()
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

    # Treina modelo final com best_params (filtrando parâmetros internos)
    # Reconstrói os parâmetros corretos para o modelo final
    best_trial = study.best_trial
    n_layers = best_trial.params["n_layers"]
    hidden_layer_sizes = []
    for i in range(n_layers):
        layer_size = best_trial.params[f"layer_{i}_size"]
        hidden_layer_sizes.append(layer_size)
    
    final_params = {
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "activation": best_trial.params["activation"],
        "alpha": best_trial.params["alpha"],
        "learning_rate": best_trial.params["learning_rate"],
        "max_iter": best_trial.params["max_iter"]
    }
    
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(random_state=30, **final_params))
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
            'optimization_time': total_end - total_start
        }
        save_results_to_file('mlp', results)
    
    print(f"MLP - Melhor acurácia (CV): {study.best_value:.4f}")
    print(f"MLP - Acurácia no teste: {test_metrics['accuracy']:.4f}")

    return final_pipeline


def optimize_svc_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para SVC usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []

    # Validação cruzada estratificada interna (5-fold)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        }
        
        # Adicionar parâmetros específicos do kernel
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        if kernel in ["poly", "rbf", "sigmoid"]:
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(random_state=30, **params))
        ])

        start = time.time()
        score = cross_val_score(
            pipeline, X_trainval, y_trainval,
            cv=inner_cv,
            scoring="accuracy"
        ).mean()
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

    # Treina modelo final com best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(random_state=30, **study.best_params))
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
            'optimization_time': total_end - total_start
        }
        save_results_to_file('svc', results)
    
    print(f"SVC - Melhor acurácia (CV): {study.best_value:.4f}")
    print(f"SVC - Acurácia no teste: {test_metrics['accuracy']:.4f}")

    return final_pipeline
