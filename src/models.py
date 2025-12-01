
"""
Módulo para otimização de hiperparâmetros de modelos de machine learning.
Contém funções de otimização usando Optuna para diferentes algoritmos de classificação.
Implementa validação cruzada aninhada (nested cross-validation) para avaliação não-enviesada.
"""

from dataclasses import dataclass
from typing import Dict, List, Any

import time
import numpy as np
import optuna
import optunahub
import os
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
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from src.reports import save_detailed_results_txt_by_fold, generate_experiment_folder_name, save_holdout_results


IMBALANCE_RATIO = 93.0/7.0
THREADS = 12-1
    
@dataclass
class FoldMetrics:
    """Data class to store metrics for a single fold"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float

@dataclass
class FoldResults:
    """Data class to store complete results for a single fold"""
    fold: int
    best_params: Dict[str, Any]
    params_importances: Dict[str, float]
    train_metrics: FoldMetrics
    test_metrics: FoldMetrics
    trials: List[Dict[str, Any]]
    best_trial_number: int = None
    test_predictions: dict = None

def balance_fold(X_train, y_train, balance_strategy):
    if not balance_strategy or balance_strategy == 'none':
        return X_train, y_train
    if balance_strategy == 'smote':
        from imblearn.over_sampling import SMOTE
        balancer = SMOTE(random_state=42)
    elif balance_strategy == 'adasyn':
        from imblearn.over_sampling import ADASYN
        balancer = ADASYN(random_state=42)
    elif balance_strategy == 'randomundersampler':
        from imblearn.under_sampling import RandomUnderSampler
        balancer = RandomUnderSampler(random_state=42)
    elif balance_strategy == 'smoteenn':
        from imblearn.combine import SMOTEENN
        balancer = SMOTEENN(random_state=42)
    elif balance_strategy == 'tomeklinks':
        from imblearn.under_sampling import TomekLinks
        balancer = TomekLinks(random_state=42)
    else:
        raise ValueError(f"Unknown balance_strategy: {balance_strategy}")
    print("Balance strategy:", balance_strategy)
    return balancer.fit_resample(X_train, y_train)

def create_objective_function(classifier_class, param_suggestions_func, custom_params_processor, 
                            fixed_params, classification_type, X_train, y_train):
    """Create the objective function for Optuna optimization"""
    def objective(trial):
        # Get parameter suggestions
        if param_suggestions_func.__name__ in ['_suggest_catboost_params', '_suggest_catboost_params_less', 
                                                '_suggest_xgboost_params', '_suggest_xgboost_params_less']:
            params = param_suggestions_func(trial, classification_type)
        else:
            params = param_suggestions_func(trial)

        # Apply custom processing if provided
        if custom_params_processor:
            params = custom_params_processor(params)

        # Add fixed parameters (excluding balance_strategy which is not a model parameter)
        if fixed_params:
            model_params = {k: v for k, v in fixed_params.items() if k != 'balance_strategy'}
            params.update(model_params)

        # Create and evaluate pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier_class(**params))
        ])

        # Internal cross-validation with balancing inside each fold
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        balance_strategy = fixed_params.get('balance_strategy') if fixed_params and 'balance_strategy' in fixed_params else None
        scores = []
        for train_idx, test_idx in inner_cv.split(X_train, y_train):
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]
            # Apply balancing to training fold only
            X_tr_bal, y_tr_bal = balance_fold(X_tr, y_tr, balance_strategy)
            pipeline.fit(X_tr_bal, y_tr_bal)
            y_pred_proba = pipeline.predict_proba(X_te)
            if classification_type == "multiclass":
                score = average_precision_score(y_te, y_pred_proba)
            else:
                score = average_precision_score(y_te, y_pred_proba[:, 1])
            scores.append(score)
        return np.mean(scores)
    
    return objective

def split_data_for_fold(X, y, train_idx, test_idx):
    """Split data for a specific fold, handling both pandas and numpy arrays"""
    
    # Handle X (features)
    if hasattr(X, 'iloc'):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    else:
        X_train, X_test = X[train_idx], X[test_idx]
    
    # Handle y (target)
    if hasattr(y, 'iloc'):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def get_model_predictions(pipeline, X_data):
    """Get predictions and probabilities from a trained pipeline"""
    
    predictions = pipeline.predict(X_data)
    
    # Get probabilities safely
    try:
        probabilities = pipeline.predict_proba(X_data)
    except AttributeError:
        # Fallback to classifier if pipeline doesn't have predict_proba
        classifier = pipeline.named_steps['classifier']
        if hasattr(classifier, 'predict_proba'):
            X_scaled = pipeline.named_steps['scaler'].transform(X_data)
            probabilities = classifier.predict_proba(X_scaled)
        else:
            raise AttributeError("Model doesn't support probability predictions")
    
    return predictions, probabilities

def calculate_metrics(y_true, y_pred, y_pred_proba, classification_type):
    """Calculate evaluation metrics for predictions"""
    
    accuracy = accuracy_score(y_true, y_pred)
    
    average_type = 'macro' if classification_type == "multiclass" else 'binary'
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average_type
    )
    
    if classification_type == "multiclass":
        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        pr_auc = average_precision_score(y_true, y_pred_proba, average='macro')
    else:
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        pr_auc = average_precision_score(y_true, y_pred_proba[:, 1])
    
    return FoldMetrics(accuracy, precision, recall, f1, roc_auc, pr_auc)

def optimize_single_outer_fold(fold_number, X_train, X_test, y_train, y_test, 
                        classifier_class, param_suggestions_func, custom_params_processor,
                        fixed_params, classification_type, model_name, n_trials, 
                        data_source="ana", balance_strategy="none"):
    """Optimize hyperparameters for a single fold"""
    
    print(f"Fold {fold_number}")
    # Create objective function
    objective = create_objective_function(
        classifier_class, param_suggestions_func, custom_params_processor,
        fixed_params, classification_type, X_train, y_train
    )

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    # Create and run Optuna study
    study_name = f"{model_name}_fold_{fold_number}"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, gc_after_trial=True)

    # Get best parameters - recreate them using the suggestion function
    # to avoid including intermediate trial parameters (like n_layers for MLP)
    if param_suggestions_func.__name__ in ['_suggest_catboost_params', '_suggest_catboost_params_less', 
                                            '_suggest_xgboost_params', '_suggest_xgboost_params_less']:
        best_params = param_suggestions_func(study.best_trial, classification_type)
    else:
        best_params = param_suggestions_func(study.best_trial)

    if custom_params_processor:
        best_params = custom_params_processor(best_params)
    if fixed_params:
        # Only add model parameters, exclude balance_strategy
        model_params = {k: v for k, v in fixed_params.items() if k != 'balance_strategy'}
        best_params.update(model_params)

    # Calcular importâncias dos parâmetros após otimização
    try:
        param_importances = optuna.importance.get_param_importances(study)
        print(f"Importâncias dos parâmetros para fold {fold_number}: {param_importances}")
    except Exception as e:
        print(f"Não foi possível calcular importâncias dos parâmetros: {e}")
        param_importances = {}

    # Save Optuna visualization plots
    try:
        import optuna.visualization as vis
        from pathlib import Path
        
        # Create directory for plots inside the experiment folder
        experiment_folder = generate_experiment_folder_name(data_source, "optimized", classification_type, balance_strategy)
        experiment_dir = os.path.join("./results", experiment_folder)
        model_dir_name = model_name.lower().replace(' ', '_')
        plots_dir = Path(experiment_dir) / model_dir_name / "optuna_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # # Save optimization history
        # fig = vis.plot_optimization_history(study)
        # fig.write_html(str(plots_dir / f"{model_dir_name}_fold_{fold_number}_history.html"))
        
        # # Save parameter importances
        # fig = vis.plot_param_importances(study)
        # fig.write_html(str(plots_dir / f"{model_dir_name}_fold_{fold_number}_importances.html"))
        
        # # Save parallel coordinate plot
        # fig = vis.plot_parallel_coordinate(study)
        # fig.write_html(str(plots_dir / f"{model_dir_name}_fold_{fold_number}_parallel.html"))
        
        # # Save slice plot
        # fig = vis.plot_slice(study)
        # fig.write_html(str(plots_dir / f"{model_dir_name}_fold_{fold_number}_slice.html"))
        
        # # Save contour plot
        # fig = vis.plot_contour(study)
        # fig.write_html(str(plots_dir / f"{model_dir_name}_fold_{fold_number}_contour.html"))
        
        print(f"Saved Optuna plots to {plots_dir}")
    except Exception as e:
        print(f"Could not save Optuna plots: {e}")

    # Train final model for this fold training set
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier_class(**best_params))
    ])
    final_pipeline.fit(X_train, y_train)

    # Evaluate on both training and test sets of one outer fold
    y_pred_train, y_pred_proba_train = get_model_predictions(final_pipeline, X_train)
    y_pred_test, y_pred_proba_test = get_model_predictions(final_pipeline, X_test)

    train_metrics = calculate_metrics(y_train, y_pred_train, y_pred_proba_train, classification_type)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test, classification_type)

    # Save test predictions for plotting
    test_predictions = {
        'y_true': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
        'y_pred': y_pred_test.tolist() if hasattr(y_pred_test, 'tolist') else list(y_pred_test),
        'y_pred_proba': y_pred_proba_test.tolist() if hasattr(y_pred_proba_test, 'tolist') else list(y_pred_proba_test)
    }

    # Collect trial data
    trials = []
    for trial in study.trials:
        trials.append({
            'trial_number': trial.number,
            'params': trial.params,
            'score': trial.value if trial.value is not None else 0.0
        })

    return FoldResults(fold = fold_number, best_params = best_params, train_metrics = train_metrics, test_metrics = test_metrics, 
                       trials = trials, params_importances = param_importances, best_trial_number=study.best_trial.number, test_predictions=test_predictions)

def aggregate_results(fold_results: List[FoldResults]) -> Dict[str, Dict[str, float]]:
    """Aggregate results across all folds"""
    
    metrics_dict = {
        'accuracy': [fr.test_metrics.accuracy for fr in fold_results],
        'precision': [fr.test_metrics.precision for fr in fold_results],
        'recall': [fr.test_metrics.recall for fr in fold_results],
        'f1': [fr.test_metrics.f1 for fr in fold_results],
        'roc_auc': [fr.test_metrics.roc_auc for fr in fold_results]
    }
    
    aggregated = {}
    for metric, scores in metrics_dict.items():
        aggregated[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    
    return aggregated

def save_optimization_results(model_name, fold_results, 
                            data_source, classification_type, balance_strategy="none"):
    """Save optimization results to files"""
    
    # Save detailed results by fold
    experiment_folder = generate_experiment_folder_name(data_source, "optimized", classification_type, balance_strategy)
    experiment_dir = os.path.join("./results", experiment_folder)
    model_dir_name = model_name.lower().replace(' ', '_')
    model_dir = os.path.join(experiment_dir, model_dir_name)
    os.makedirs(model_dir, exist_ok=True)
    # from pprint import pprint
    # pprint(fold_results)

    
    # Convert fold_results to the format expected by save_detailed_results_txt_by_fold
    all_folds_trials = []
    
    try:
        for fr in fold_results:
            # Convert np.float64 in param_importances to native float
            param_importances_clean = {k: float(v) for k, v in fr.params_importances.items()}
            fold_data = {
                'fold': fr.fold,
                'trials': fr.trials,
                'best_params': fr.best_params,
                'param_importances': param_importances_clean,
                'best_trial_number': fr.best_trial_number,
                'train_metrics': {
                    'accuracy': fr.train_metrics.accuracy,
                    'precision': fr.train_metrics.precision,
                    'recall': fr.train_metrics.recall,
                    'f1': fr.train_metrics.f1,
                    'roc_auc': fr.train_metrics.roc_auc,
                    'pr_auc': fr.train_metrics.pr_auc
                },
                'test_metrics': {
                    'accuracy': fr.test_metrics.accuracy,
                    'precision': fr.test_metrics.precision,
                    'recall': fr.test_metrics.recall,
                    'f1': fr.test_metrics.f1,
                    'roc_auc': fr.test_metrics.roc_auc,
                    'pr_auc': fr.test_metrics.pr_auc
                },
                'test_predictions': fr.test_predictions if hasattr(fr, 'test_predictions') else None
            }
            all_folds_trials.append(fold_data)
    except Exception as e:
        print(f"Erro ao preparar dados detalhados por fold: {e}")
        all_folds_trials = []
    
    # Get best fold based on AUCPR score
    best_fold_idx = np.argmax([fr.test_metrics.pr_auc for fr in fold_results])
    final_best_params = fold_results[best_fold_idx].best_params
    
    detailed_results_path = os.path.join(model_dir, "detailed_results_by_fold.txt")
    save_detailed_results_txt_by_fold(
        model_name=model_name,
        all_folds_trials=all_folds_trials,
        output_path=detailed_results_path,
        final_best_params=final_best_params
    )
    print(f"Detailed results by fold saved to: {detailed_results_path}")

    # Save test predictions for all folds in metrics.json for plotting
    metrics_json_path = os.path.join(model_dir, "metrics.json")
    y_true_all, y_pred_all, y_pred_proba_all = [], [], []
    for fold in all_folds_trials:
        preds = fold.get('test_predictions')
        if preds:
            y_true_all.extend(preds.get('y_true', []))
            y_pred_all.extend(preds.get('y_pred', []))
            y_pred_proba_all.extend(preds.get('y_pred_proba', []))
    metrics_json = {
        'model_name': model_name,
        'test_predictions': {
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'y_pred_proba': y_pred_proba_all
        },
        'folds': len(all_folds_trials),
        'final_best_params': final_best_params
    }
    import json
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    print(f"Test predictions for all folds saved to: {metrics_json_path}")

def _optimize_classifier_generic(classifier_class, param_suggestions_func, model_name, X, y,
                               n_trials=100, custom_params_processor=None, fixed_params=None, data_source="ana",
                               classification_type="binary", outer_cv_folds=5, use_less_params=False):
    """
    Refactored version of the classifier optimization function.
    
    Args:
        use_less_params: If True, uses simplified parameter suggestion functions (ignored, handled by caller)
    """
    print(f"\nStarting Nested Cross-Validation for {model_name}...")
    print(f"Configuration: {outer_cv_folds} outer folds, {n_trials} trials per fold")
    
    # First perform 80/20 holdout split
    print("Performing 80/20 holdout split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Holdout split completed:")
    print(f"  - Training set: {X_train.shape[0]} samples (80%)")
    print(f"  - Holdout test set: {X_test.shape[0]} samples (20%)")
    
    # Set up cross-validation on the 80% training data only
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42) 
    
    # Process all folds using only the 80% training data
    fold_results_list: FoldResults = []
    fold_number = 1
    
    for train_idx, test_idx in outer_cv.split(X_train, y_train):
        # Split data for this fold from the 80% training data
        X_fold_train, X_fold_test, y_fold_train, y_fold_test = split_data_for_fold(
            X_train, y_train, train_idx, test_idx
        )
        # Optimize this fold
        fold_result = optimize_single_outer_fold(
            fold_number, X_fold_train, X_fold_test, y_fold_train, y_fold_test,
            classifier_class, param_suggestions_func, custom_params_processor,
            fixed_params, classification_type, model_name, n_trials, data_source,
            balance_strategy=fixed_params.get('balance_strategy', 'none') if fixed_params else 'none'
        )
        # Only append if not None
        if fold_result is not None:
            fold_results_list.append(fold_result)
        else:
            print(f"Fold {fold_number} returned None and was skipped.")
        fold_number += 1
    
    print(f"\nNested CV completed. The 20% holdout set ({X_test.shape[0]} samples) remains untouched.")
    print("This holdout set can be used for final unbiased evaluation.")
    
    # Now train and evaluate each of the best parameter sets on the holdout data
    print(f"\nTraining and evaluating {len(fold_results_list)} best parameter sets on holdout data...")
    
    holdout_results = []
    for i, fold_result in enumerate(fold_results_list, 1):
        print(f"Evaluating fold {i} best params on holdout set...")
        
        # Train model with best params from this fold on the full 80% training data
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier_class(**fold_result.best_params))
        ])
        final_pipeline.fit(X_train, y_train)
        
        # Evaluate on holdout set
        y_pred_holdout, y_pred_proba_holdout = get_model_predictions(final_pipeline, X_test)
        holdout_metrics = calculate_metrics(y_test, y_pred_holdout, y_pred_proba_holdout, classification_type)
        
        holdout_results.append({
            'fold': i,
            'best_params': fold_result.best_params,
            'holdout_metrics': holdout_metrics
        })

    print("Saving optimization and holdout results...")
    save_optimization_results(
        model_name, fold_results_list,
        data_source, classification_type,
        balance_strategy=fixed_params.get('balance_strategy', 'none') if fixed_params else 'none'
    )
    print("Saving holdout results...")
    save_holdout_results(
        model_name, holdout_results,
        data_source, classification_type,
        balance_strategy=fixed_params.get('balance_strategy', 'none') if fixed_params else 'none'
    )
    print("Saved!")

    # Return best model and test metrics for compatibility
    # Select best fold by pr_auc
    if fold_results_list:
        best_fold_idx = np.argmax([fr.test_metrics.pr_auc for fr in fold_results_list])
        best_fold = fold_results_list[best_fold_idx]
        # Train model with best params from best fold on all training data
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier_class(**best_fold.best_params))
        ])
        best_model.fit(X_train, y_train)
        # Evaluate on holdout set
        y_pred_holdout, y_pred_proba_holdout = get_model_predictions(best_model, X_test)
        test_metrics = calculate_metrics(y_test, y_pred_holdout, y_pred_proba_holdout, classification_type)
        return best_model, test_metrics
    else:
        return None, None
    

# Funções de sugestão de parâmetros para cada modelo #


def _suggest_catboost_params(trial, classification_type="binary"):
    """Sugestões de parâmetros para CatBoost"""
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "verbose": False,  # Silenciar logs durante otimização
        "allow_writing_files": False,  # Não escrever arquivos de log
        "thread_count": THREADS,
    }
    
    # Configure loss function and weights based on classification type
    if classification_type == "multiclass":
        params["loss_function"] = "MultiClass"
        params["classes_count"] = 3  # TSG=1, Oncogene=2, Passenger=0
        # Don't use scale_pos_weight for multiclass
    else:
        params["loss_function"] = "Logloss"
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", IMBALANCE_RATIO * 0.4, IMBALANCE_RATIO * 4)
    
    return params


def _suggest_catboost_params_less(trial, classification_type="binary"):
    """Sugestões de parâmetros reduzidos para CatBoost"""
    params = {
        "iterations": trial.suggest_int("iterations", 50, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),

        # "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
        # "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.02),
        # "depth": trial.suggest_int("depth", 2, 4),
        # "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        # "border_count": trial.suggest_int("border_count", 150, 255),
        # "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1.0),
        # "random_strength": trial.suggest_float("random_strength", 0, 1.0),

        "verbose": False,
        "allow_writing_files": False,
        "thread_count": THREADS,
    }
    
    # Configure loss function and weights based on classification type
    if classification_type == "multiclass":
        params["loss_function"] = "MultiClass"
        params["classes_count"] = 3  # TSG=1, Oncogene=2, Passenger=0
    else:
        params["scale_pos_weight"] = IMBALANCE_RATIO
        params["loss_function"] = "Logloss"
        # params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 5.0,6.0)
    
    return params


def _suggest_decision_tree_params(trial):
    """Sugestões de parâmetros para Decision Tree"""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "splitter": trial.suggest_categorical("splitter", ["best", "random"])
    }


def _suggest_decision_tree_params_less(trial):
    """Sugestões de parâmetros reduzidos para Decision Tree"""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),

        # "max_depth": trial.suggest_int("max_depth", 2, 32),
        # "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
        # "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 20),
        # "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        # "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        # "splitter": trial.suggest_categorical("splitter", ["best", "random"])


    }


def _suggest_gradient_boosting_params(trial):
    """Sugestões de parâmetros para Gradient Boosting"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
    }


def _suggest_gradient_boosting_params_less(trial):
    """Sugestões de parâmetros reduzidos para Gradient Boosting"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 30)

        # "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        # "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.1),
        # "max_depth": trial.suggest_int("max_depth", 2, 8),
        # "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        # "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        # "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        # "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"])

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


def _suggest_hist_gradient_boosting_params_less(trial):
    """Sugestões de parâmetros reduzidos para Histogram Gradient Boosting"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 30)

        # "max_iter": trial.suggest_int("max_iter", 50, 200),
        # "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.1),
        # "max_depth": trial.suggest_int("max_depth", 2, 5),
        # "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        # "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0)

    }


def _suggest_knn_params(trial):
    """Sugestões de parâmetros para KNN"""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "p": trial.suggest_int("p", 1, 2),  # 1 for manhattan, 2 for euclidean
        "leaf_size": trial.suggest_int("leaf_size", 20, 40)
    }


def _suggest_knn_params_less(trial):
    """Sugestões de parâmetros reduzidos para KNN"""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
        "p": trial.suggest_int("p", 1, 4),  # 1 for manhattan, 2 for euclidean

        # "n_neighbors": trial.suggest_int("n_neighbors", 25, 125, step=10),
        # "weights": "distance",
        # "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        # "p": trial.suggest_int("p", 1, 2),  # 1 for manhattan, 2 for euclidean
        # "leaf_size": trial.suggest_int("leaf_size", 20, 40)
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
        "solver": trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"]),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1000)
    }


def _suggest_mlp_params_less(trial):
    """Sugestões de parâmetros reduzidos para MLP"""
    n_layers = trial.suggest_int("n_layers", 1, 2)
    hidden_layer_sizes = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 100, 200, step=25)
        hidden_layer_sizes.append(layer_size)
    
    return {
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "activation": trial.suggest_categorical("activation", ["tanh", "logistic"]),
        "alpha": trial.suggest_float("alpha", 1e-4, 0.01, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "solver": trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"]),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1500)
    }


def _suggest_random_forest_params(trial):
    """Sugestões de parâmetros para Random Forest"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "n_jobs": -1
    }


def _suggest_random_forest_params_less(trial):
    """Sugestões de parâmetros reduzidos para Random Forest"""
    return {
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7]),
        "max_depth": trial.suggest_int("max_depth", 5, 50),

        # "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=25),
        # "max_depth": trial.suggest_int("max_depth", 3, 20),
        # "min_samples_split": trial.suggest_int("min_samples_split", 5, 20, step=2),
        # "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 25, step=2),
        # "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        # "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        # "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),

        "n_jobs": -1
    }


def _suggest_svc_params(trial):
    """Sugestões de parâmetros para SVC otimizadas para evitar execução infinita"""
    
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 0.1, 100.0, log=True), 
            "probability": True, 
            "max_iter": 1000,  
            "tol": 1e-3,  
            "cache_size": 200,  
            "degree": trial.suggest_int("degree", 2, 6),
            "shrinking": trial.suggest_categorical("shrinking", [True, False])
    }
    
    if kernel == "rbf":
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    return params


def _suggest_svc_params_less(trial):
    """Sugestões de parâmetros reduzidos para SVC"""
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    params = {
        "kernel": kernel,
    }
    
    return params


def _suggest_xgboost_params(trial, classification_type="binary"):
    """Sugestões de parâmetros para XGBoost"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,  # Use all available cores
        "verbosity": 0,  # Silenciar logs durante otimização
    }
    
    # Configure objective and weights based on classification type
    if classification_type == "multiclass":
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        # Don't use scale_pos_weight for multiclass
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", IMBALANCE_RATIO * 0.4, IMBALANCE_RATIO * 4)
    
    return params


def _suggest_xgboost_params_less(trial, classification_type="binary"):
    """Sugestões de parâmetros reduzidos para XGBoost"""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 30),

        # "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step = 50),
        # "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.1),
        # "max_depth": trial.suggest_int("max_depth", 2, 20, step = 2),
        # "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        # "subsample": trial.suggest_float("subsample", 0.9, 1),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.6),
        # "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        # "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        # "gamma": trial.suggest_float("gamma", 0.0, 5.0, step=1.0),
        # "random_state": 42,
        # "n_jobs": -1,  # Use all available cores
        # "verbosity": 0,  # Silenciar logs durante otimização

        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    
    # Configure objective and weights based on classification type
    if classification_type == "multiclass":
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        #params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 5.0,6.0)
        params["scale_pos_weight"] = IMBALANCE_RATIO
    
    return params


def optimize_decision_tree_classifier(X, y, n_trials=30, save_results=True, fixed_params=None, 
                                    data_source="ana", classification_type="binary", 
                                    use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para Decision Tree Classifier usando Optuna"""
    if fixed_params is None:
        fixed_params = {}
    fixed_params["class_weight"] = "balanced"
    param_func = _suggest_decision_tree_params_less if use_less_params else _suggest_decision_tree_params
    return _optimize_classifier_generic(
        DecisionTreeClassifier,
        param_func,
        'decision_tree',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )


def optimize_random_forest_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                                     data_source="ana", classification_type="binary", 
                                     use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para Random Forest Classifier usando Optuna"""
    if fixed_params is None:
        fixed_params = {}
    fixed_params["class_weight"] = "balanced"
    param_func = _suggest_random_forest_params_less if use_less_params else _suggest_random_forest_params
    return _optimize_classifier_generic(
        RandomForestClassifier,
        param_func,
        'random_forest',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )


def optimize_gradient_boosting_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                                        data_source="ana", classification_type="binary", 
                                        use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para Gradient Boosting Classifier usando Optuna"""
    param_func = _suggest_gradient_boosting_params_less if use_less_params else _suggest_gradient_boosting_params
    return _optimize_classifier_generic(
        GradientBoostingClassifier,
        param_func,
        'gradient_boosting',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )


def optimize_hist_gradient_boosting_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                                             data_source="ana", classification_type="binary", 
                                             use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para Histogram Gradient Boosting Classifier usando Optuna"""
    param_func = _suggest_hist_gradient_boosting_params_less if use_less_params else _suggest_hist_gradient_boosting_params
    return _optimize_classifier_generic(
        HistGradientBoostingClassifier,
        param_func,
        'histogram_gradient_boosting',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )


def optimize_knn_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                           data_source="ana", classification_type="binary", 
                           use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para KNN Classifier usando Optuna"""
    param_func = _suggest_knn_params_less if use_less_params else _suggest_knn_params
    return _optimize_classifier_generic(
        KNeighborsClassifier,
        param_func,
        'k_nearest_neighbors',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )


def optimize_mlp_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                          data_source="ana", classification_type="binary", 
                          use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para MLP Classifier usando Optuna"""
    param_func = _suggest_mlp_params_less if use_less_params else _suggest_mlp_params
    return _optimize_classifier_generic(
        MLPClassifier,
        param_func,
        'multi_layer_perceptron',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )


def optimize_svc_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                          data_source="ana", classification_type="binary", 
                          use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para SVC usando Optuna"""
    enforced_params = {"probability": True, "class_weight": "balanced"}
    if fixed_params:
        enforced_params.update(fixed_params)
    param_func = _suggest_svc_params_less if use_less_params else _suggest_svc_params
    return _optimize_classifier_generic(
        SVC,
        param_func,
        'svc',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=enforced_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )


def optimize_catboost_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                               data_source="ana", classification_type="binary", 
                               use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para CatBoost Classifier usando Optuna"""
    param_func = _suggest_catboost_params_less if use_less_params else _suggest_catboost_params
    return _optimize_classifier_generic(
        CatBoostClassifier,
        param_func,
        'catboost',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )


def optimize_xgboost_classifier(X, y, n_trials=30, save_results=True, fixed_params=None,
                              data_source="ana", classification_type="binary", 
                              use_nested_cv=True, outer_cv_folds=5, use_less_params=False):
    """Otimização de hiperparâmetros para XGBoost Classifier usando Optuna"""
    param_func = _suggest_xgboost_params_less if use_less_params else _suggest_xgboost_params
    return _optimize_classifier_generic(
        XGBClassifier,
        param_func,
        'xgboost',
        X, y, n_trials,
        custom_params_processor=None,
        fixed_params=fixed_params,
        data_source=data_source, classification_type=classification_type,
        outer_cv_folds=outer_cv_folds, use_less_params=use_less_params
    )
