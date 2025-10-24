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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from src.evaluation import detailed_cross_val_score, evaluate_classification_on_test
from src.reports import save_model_results_unified


def _nested_cross_validation_optimization(classifier_class, param_suggestions_func, model_name, X, y, 
                                       n_trials=100, save_results=True, custom_params_processor=None, 
                                       return_test_metrics=False, fixed_params=None, data_source="ana",
                                       classification_type="binary", outer_cv_folds=5):
    """
    Implementa nested cross-validation para avaliação imparcial do modelo.
    
    O loop externo divide os dados em treino/teste para avaliação final.
    O loop interno otimiza hiperparâmetros usando cross-validation no conjunto de treino.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    
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
        
        # Dividir dados para este fold
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
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
                                   cv=inner_cv, scoring='f1_weighted' if classification_type == "multiclass" else 'f1')
            
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
        y_pred_proba_fold = final_pipeline.predict_proba(X_test_fold)
        
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
        from .reports import save_nested_cv_results
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


def _simple_holdout_optimization(classifier_class, param_suggestions_func, model_name, X, y, 
                               n_trials=100, save_results=True, custom_params_processor=None, 
                               return_test_metrics=False, fixed_params=None, data_source="ana",
                               classification_type="binary"):
    """
    Implementa otimização simples com holdout (train/test split).
    Mantém a funcionalidade original para compatibilidade.
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
        # Use fixed params if provided, otherwise suggest params
        if fixed_params:
            params = fixed_params.copy()
        else:
            params = param_suggestions_func(trial)

        # Only pass random_state to classifiers that support it
        classifier_kwargs = params.copy()
        if classifier_class not in [KNeighborsClassifier]:
            classifier_kwargs['random_state'] = 30
            
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier_class(**classifier_kwargs))
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

    # Otimização com Optuna ou execução única com parâmetros fixos
    if fixed_params:
        # Execução única com parâmetros fixos
        study = optuna.create_study(direction="maximize")
        total_start = time.time()
        study.optimize(objective, n_trials=1)
        total_end = time.time()
        final_params = fixed_params.copy()
        best_score = study.best_value
    else:
        # Otimização normal
        study = optuna.create_study(direction="maximize")
        total_start = time.time()
        study.optimize(objective, n_trials=n_trials)
        total_end = time.time()
        
        # Processar parâmetros se necessário (para casos especiais como MLP)
        final_params = study.best_params.copy()
        if custom_params_processor:
            final_params = custom_params_processor(study.best_trial)
        best_score = study.best_value
    
    # Para SVC, garantir que probability=True esteja sempre presente
    if classifier_class == SVC:
        final_params["probability"] = True

    # Treina modelo final com best_params
    # Only pass random_state to classifiers that support it
    final_classifier_kwargs = final_params.copy()
    if classifier_class not in [KNeighborsClassifier]:
        final_classifier_kwargs['random_state'] = 30
        
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier_class(**final_classifier_kwargs))
    ])
    final_pipeline.fit(X_trainval, y_trainval)

    # Avalia no conjunto de teste
    test_metrics = evaluate_classification_on_test(final_pipeline, X_test, y_test, return_dict=True, classification_type=classification_type)

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
            
        save_model_results_unified(model_name, results, mode="optimized",
                                 data_source=data_source, classification_type=classification_type)
    
    print(f"{model_name.replace('_', ' ').title()} - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"{model_name.replace('_', ' ').title()} - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"{model_name.replace('_', ' ').title()} - PR AUC no teste: {test_metrics['pr_auc']:.4f}")
    if model_name == 'decision_tree':
        print(f"{model_name.replace('_', ' ').title()} - CV folds utilizados: {n_splits}")

    if return_test_metrics:
        return final_pipeline, test_metrics
    else:
        return final_pipeline


def _optimize_classifier_generic(classifier_class, param_suggestions_func, model_name, X, y, 
                               n_trials=100, save_results=True, custom_params_processor=None, 
                               return_test_metrics=False, fixed_params=None, data_source="ana",
                               classification_type="binary", use_nested_cv=True, outer_cv_folds=5):
    """
    Função genérica para otimização de hiperparâmetros de classificadores com nested cross-validation.
    
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
    return_test_metrics : bool
        Se deve retornar também as métricas de teste
    fixed_params : dict, optional
        Parâmetros fixos para o modelo
    data_source : str
        Fonte dos dados ("ana" ou "renan")
    classification_type : str
        Tipo de classificação ("binary" ou "multiclass")
    use_nested_cv : bool
        Se deve usar nested cross-validation (True) ou holdout simples (False)
    outer_cv_folds : int
        Número de folds para o loop externo do nested CV
        
    Returns:
    --------
    sklearn.pipeline.Pipeline ou tuple
        Pipeline otimizado, ou tuple (pipeline, test_metrics) se return_test_metrics=True
    """
    
    if use_nested_cv:
        return _nested_cross_validation_optimization(
            classifier_class, param_suggestions_func, model_name, X, y,
            n_trials, save_results, custom_params_processor, return_test_metrics, 
            fixed_params, data_source, classification_type, outer_cv_folds
        )
    else:
        return _simple_holdout_optimization(
            classifier_class, param_suggestions_func, model_name, X, y,
            n_trials, save_results, custom_params_processor, return_test_metrics,
            fixed_params, data_source, classification_type
        )


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
    return _optimize_classifier_generic(
        SVC,
        _suggest_svc_params,
        'svc',
        X, y, n_trials, save_results, 
        custom_params_processor=None,
        return_test_metrics=True,
        fixed_params=fixed_params,
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
