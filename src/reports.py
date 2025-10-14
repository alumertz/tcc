"""
Módulo para geração de relatórios e salvamento de resultados de modelos de machine learning.
Contém funções para formatação de resultados de validação cruzada, relatórios de classificação e persistência.
"""

import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)


def generate_enhanced_classification_report(y_true, y_pred, y_pred_proba):
    """
    Gera relatório de classificação customizado incluindo accuracy, ROC AUC e PR AUC
    """
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


def save_results_to_file(model_name, results, results_dir="results/omics"):
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

        # ALTERADO À MÃO
        f.write("MULTIÔMICAS UTILIZADAS:\n")
        f.write("-"*30 + "\n")
        f.write("Mutations")
        f.write("\n\n")
        
        f.write("HISTÓRICO DE OTIMIZAÇÃO:\n")
        f.write("-"*30 + "\n")
        f.write(f"Número de trials: {len(results['trials'])}\n")
        f.write(f"Melhor score (CV): {results['best_score']:.4f}\n")
        f.write(f"Tempo total de otimização: {results['optimization_time']:.2f} segundos\n")
    
    print(f"Resultados salvos em: {model_dir}")
    return trials_file, test_results_file


def save_simple_results_to_file(model_name, results, results_dir="results/omics"):
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar uma cópia filtrada do results para salvar no JSON (sem cv_metrics e test_metrics)
    results_for_json = {k: v for k, v in results.items() if k not in ['cv_metrics', 'test_metrics']}
    
    # Salvar JSON com os dados filtrados
    json_file = os.path.join(model_dir, f"results_{timestamp}.json")
    with open(json_file, 'w') as jf:
        json.dump(results_for_json, jf, indent=2, default=str)
    
    # Salvar arquivo de texto com resumo completo (incluindo métricas)
    test_results_file = os.path.join(model_dir, f"test_results_{timestamp}.txt")
    with open(test_results_file, 'w') as f:
        f.write(f"RESULTADOS DO MODELO: {model_name.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if 'all_cv_metrics' in results:
            generate_all_trials_cv_tables(f, results['all_cv_metrics'])
        elif 'cv_detailed_metrics' in results:
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
        
        f.write("MULTIÔMICAS UTILIZADAS:\n")
        f.write("-"*30 + "\n")
        for omic in results.get('omics_used', []):
            f.write(f"{omic}\n")
        f.write("\n")

    print(f"Resultados JSON salvos em: {json_file}")
    print(f"Resumo de resultados salvos em: {test_results_file}")
    return json_file, test_results_file