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
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Gerar tabelas de métricas de CV para todos os trials se disponível
        if 'all_cv_metrics' in results:
            generate_all_trials_cv_tables(f, results['all_cv_metrics'])
        elif 'cv_detailed_metrics' in results:
            # Fallback para compatibilidade com versão anterior
            generate_cv_metrics_table(f, results['cv_detailed_metrics'])
        
        f.write("AVALIAÇÃO NO CONJUNTO DE TESTE FINAL:\n")
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


def generate_cv_metrics_table(file_handle, cv_metrics):
    """
    Gera tabela formatada com métricas de todos os folds do CV
    """
    file_handle.write("MÉTRICAS DETALHADAS DE VALIDAÇÃO CRUZADA (5-FOLD):\n")
    file_handle.write("="*80 + "\n\n")
    
    # Extrair métricas por fold
    folds_data = cv_metrics
    
    # Nomes das métricas
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC']
    
    # Cabeçalho da tabela
    header = "Métrica".ljust(12)
    for i in range(5):  # 5 folds
        header += f"Fold{i+1}_Train".rjust(12) + f"Fold{i+1}_Val".rjust(12)
    header += "Média_Train".rjust(12) + "Média_Val".rjust(12)
    
    file_handle.write(header + "\n")
    file_handle.write("-" * len(header) + "\n")
    
    # Linhas das métricas
    for metric, metric_name in zip(metrics, metric_names):
        line = metric_name.ljust(12)
        
        train_values = []
        val_values = []
        
        # Valores para cada fold
        for fold_idx in range(5):
            if fold_idx < len(folds_data):
                train_val = folds_data[fold_idx].get(f'train_{metric}', 0.0)
                val_val = folds_data[fold_idx].get(f'val_{metric}', 0.0)
                
                line += f"{train_val:.4f}".rjust(12)
                line += f"{val_val:.4f}".rjust(12)
                
                train_values.append(train_val)
                val_values.append(val_val)
            else:
                line += "N/A".rjust(12) + "N/A".rjust(12)
        
        # Médias
        if train_values:
            avg_train = sum(train_values) / len(train_values)
            line += f"{avg_train:.4f}".rjust(12)
        else:
            line += "N/A".rjust(12)
            
        if val_values:
            avg_val = sum(val_values) / len(val_values)
            line += f"{avg_val:.4f}".rjust(12)
        else:
            line += "N/A".rjust(12)
        
        file_handle.write(line + "\n")
    
    file_handle.write("-" * len(header) + "\n")
    file_handle.write(f"Validação cruzada com {len(folds_data)} folds\n\n")


def generate_all_trials_cv_tables(file_handle, all_cv_metrics):
    """
    Gera tabelas formatadas com métricas de CV para todos os trials
    """
    file_handle.write("MÉTRICAS DETALHADAS DE VALIDAÇÃO CRUZADA - TODOS OS TRIALS:\n")
    file_handle.write("="*80 + "\n\n")
    
    # Ordenar por score (melhor primeiro)
    sorted_trials = sorted(all_cv_metrics, key=lambda x: x['score'], reverse=True)
    
    for idx, trial_data in enumerate(sorted_trials):
        trial_num = trial_data['trial_number']
        score = trial_data['score']
        params = trial_data['params']
        cv_metrics = trial_data['cv_metrics']
        
        # Cabeçalho do trial
        file_handle.write(f"TRIAL {trial_num + 1} (Rank #{idx + 1}) - PR AUC: {score:.4f}\n")
        file_handle.write("-"*60 + "\n")
        
        # Hiperparâmetros
        file_handle.write("Hiperparâmetros:\n")
        for param, value in params.items():
            file_handle.write(f"  {param}: {value}\n")
        file_handle.write("\n")
        
        # Tabela de métricas se disponível
        if cv_metrics:
            generate_single_trial_cv_table(file_handle, cv_metrics)
        else:
            file_handle.write("Métricas de CV não disponíveis para este trial.\n")
        
        file_handle.write("\n" + "="*80 + "\n\n")


def generate_single_trial_cv_table(file_handle, cv_metrics):
    """
    Gera tabela de métricas de CV para um único trial
    """
    if not cv_metrics:
        file_handle.write("Nenhuma métrica de CV disponível.\n")
        return
    
    # Nomes das métricas
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC']
    
    # Cabeçalho da tabela
    header = "Métrica".ljust(12)
    for i in range(len(cv_metrics)):  # Para cada fold
        header += f"Fold{i+1}_Train".rjust(12) + f"Fold{i+1}_Val".rjust(12)
    header += "Média_Train".rjust(12) + "Média_Val".rjust(12)
    
    file_handle.write(header + "\n")
    file_handle.write("-" * len(header) + "\n")
    
    # Linhas das métricas
    for metric, metric_name in zip(metrics, metric_names):
        line = metric_name.ljust(12)
        
        train_values = []
        val_values = []
        
        # Valores para cada fold
        for fold_idx, fold_data in enumerate(cv_metrics):
            train_val = fold_data.get(f'train_{metric}', 0.0)
            val_val = fold_data.get(f'val_{metric}', 0.0)
            
            line += f"{train_val:.4f}".rjust(12)
            line += f"{val_val:.4f}".rjust(12)
            
            train_values.append(train_val)
            val_values.append(val_val)
        
        # Médias
        if train_values:
            avg_train = sum(train_values) / len(train_values)
            line += f"{avg_train:.4f}".rjust(12)
        else:
            line += "N/A".rjust(12)
            
        if val_values:
            avg_val = sum(val_values) / len(val_values)
            line += f"{avg_val:.4f}".rjust(12)
        else:
            line += "N/A".rjust(12)
        
        file_handle.write(line + "\n")
    
    file_handle.write("-" * len(header) + "\n")
    file_handle.write(f"Validação cruzada com {len(cv_metrics)} folds\n")


def detailed_cross_val_score(pipeline, X, y, cv, scoring='average_precision'):
    """
    Executa validação cruzada coletando métricas detalhadas de cada fold
    """
    from sklearn.model_selection import cross_validate
    
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


def generate_enhanced_classification_report(y_true, y_pred, y_pred_proba):
    """
    Gera relatório de classificação customizado incluindo accuracy, ROC AUC e PR AUC
    """
    from sklearn.metrics import precision_recall_fscore_support
    import numpy as np
    
    # Calcular métricas por classe
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Métricas globais
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Cabeçalho expandido
    report = "              precision    recall  f1-score   support   accuracy   roc_auc    pr_auc\n\n"
    
    # Métricas por classe
    for i in range(len(precision_per_class)):
        class_accuracy = accuracy if i == 1 else accuracy  # Accuracy é global
        class_roc_auc = roc_auc if i == 1 else "-"  # ROC AUC só para classe positiva
        class_pr_auc = pr_auc if i == 1 else "-"    # PR AUC só para classe positiva
        
        if isinstance(class_roc_auc, str):
            report += f"           {i}       {precision_per_class[i]:.2f}      {recall_per_class[i]:.2f}      {f1_per_class[i]:.2f}      {support_per_class[i]:4d}        -         -         -\n"
        else:
            report += f"           {i}       {precision_per_class[i]:.2f}      {recall_per_class[i]:.2f}      {f1_per_class[i]:.2f}      {support_per_class[i]:4d}     {class_accuracy:.2f}      {class_roc_auc:.2f}     {class_pr_auc:.2f}\n"
    
    report += "\n"
    
    # Accuracy global com todas as métricas
    total_support = np.sum(support_per_class)
    report += f"    accuracy                           {accuracy:.2f}      {total_support:4d}     {accuracy:.2f}     {roc_auc:.2f}     {pr_auc:.2f}\n"
    
    # Macro average com todas as métricas
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    report += f"   macro avg       {macro_precision:.2f}      {macro_recall:.2f}      {macro_f1:.2f}      {total_support:4d}     {accuracy:.2f}     {roc_auc:.2f}     {pr_auc:.2f}\n"
    
    # Weighted average com todas as métricas
    weighted_precision = np.average(precision_per_class, weights=support_per_class)
    weighted_recall = np.average(recall_per_class, weights=support_per_class)
    weighted_f1 = np.average(f1_per_class, weights=support_per_class)
    report += f"weighted avg       {weighted_precision:.2f}      {weighted_recall:.2f}      {weighted_f1:.2f}      {total_support:4d}     {accuracy:.2f}     {roc_auc:.2f}     {pr_auc:.2f}\n"
    
    return report


def optimize_decision_tree_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Decision Tree Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []  # Armazenar métricas de CV de todos os trials

    # Validação cruzada estratificada interna com 5 folds
    n_splits = 5
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
                
        except ValueError as e:
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
            'cv_folds_used': n_splits,
            'all_cv_metrics': all_cv_metrics  # Todas as métricas de CV de todos os trials
        }
        save_results_to_file('decision_tree', results)
    
    print(f"Decision Tree - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"Decision Tree - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"Decision Tree - PR AUC no teste: {test_metrics['pr_auc']:.4f}")
    print(f"Decision Tree - CV folds utilizados: {n_splits}")

    return final_pipeline


def optimize_random_forest_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Random Forest Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []  # Armazenar métricas de CV de todos os trials

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
            'optimization_time': total_end - total_start,
            'all_cv_metrics': all_cv_metrics  # Todas as métricas de CV de todos os trials
        }
        save_results_to_file('random_forest', results)
    
    print(f"Random Forest - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"Random Forest - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"Random Forest - PR AUC no teste: {test_metrics['pr_auc']:.4f}")

    return final_pipeline


def optimize_gradient_boosting_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Gradient Boosting Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []  # Armazenar métricas de CV de todos os trials

    # Validação cruzada estratificada interna (5-fold)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0)
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(random_state=30, **params))
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
            'optimization_time': total_end - total_start,
            'all_cv_metrics': all_cv_metrics  # Todas as métricas de CV de todos os trials
        }
        save_results_to_file('gradient_boosting', results)
    
    print(f"Gradient Boosting - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"Gradient Boosting - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"Gradient Boosting - PR AUC no teste: {test_metrics['pr_auc']:.4f}")

    return final_pipeline


def optimize_hist_gradient_boosting_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para Histogram Gradient Boosting Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []  # Armazenar métricas de CV de todos os trials

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
            'optimization_time': total_end - total_start,
            'all_cv_metrics': all_cv_metrics  # Todas as métricas de CV de todos os trials
        }
        save_results_to_file('hist_gradient_boosting', results)
    
    print(f"Hist Gradient Boosting - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"Hist Gradient Boosting - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"Hist Gradient Boosting - PR AUC no teste: {test_metrics['pr_auc']:.4f}")

    return final_pipeline


def optimize_knn_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para KNN Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []  # Armazenar métricas de CV de todos os trials

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
            'optimization_time': total_end - total_start,
            'all_cv_metrics': all_cv_metrics  # Todas as métricas de CV de todos os trials
        }
        save_results_to_file('knn', results)
    
    print(f"KNN - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"KNN - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"KNN - PR AUC no teste: {test_metrics['pr_auc']:.4f}")

    return final_pipeline


def optimize_mlp_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para MLP Classifier usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []  # Armazenar métricas de CV de todos os trials

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
            'optimization_time': total_end - total_start,
            'all_cv_metrics': all_cv_metrics  # Todas as métricas de CV de todos os trials
        }
        save_results_to_file('mlp', results)
    
    print(f"MLP - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"MLP - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"MLP - PR AUC no teste: {test_metrics['pr_auc']:.4f}")

    return final_pipeline


def optimize_svc_classifier(X, y, n_trials=30, save_results=True):
    """Otimização de hiperparâmetros para SVC usando Optuna"""
    # Holdout para teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trial_data = []
    all_cv_metrics = []  # Armazenar métricas de CV de todos os trials

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
            ("classifier", SVC(random_state=30, probability=True, **params))
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

    # Treina modelo final com best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(random_state=30, probability=True, **study.best_params))
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
            'all_cv_metrics': all_cv_metrics  # Todas as métricas de CV de todos os trials
        }
        save_results_to_file('svc', results)
    
    print(f"SVC - Melhor PR AUC (CV): {study.best_value:.4f}")
    print(f"SVC - Acurácia no teste: {test_metrics['accuracy']:.4f}")
    print(f"SVC - PR AUC no teste: {test_metrics['pr_auc']:.4f}")

    return final_pipeline
