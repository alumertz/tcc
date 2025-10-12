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
    Gera relatório de classificação customizado com métricas completas
    """
    # Calcular métricas por classe
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Métricas globais
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Total support for summary rows
    total_support = np.sum(support_per_class)
    
    # Cabeçalho da tabela
    report = "              accuracy   precision    recall    f1-score   roc_auc    pr_auc\n\n"
    
    # Métricas por classe (Non-driver = classe 0, Driver = classe 1)
    class_names = ['Non-driver', 'Driver']
    
    for i in range(len(precision_per_class)):
        class_name = class_names[i] if i < len(class_names) else f'Class {i}'
        
        # Para a classe Driver (1), incluir ROC AUC e PR AUC, para Non-driver usar "-"
        if i == 1:  # Driver class
            report += f"{class_name:>12}       {accuracy:.4f}      {precision_per_class[i]:.4f}     {recall_per_class[i]:.4f}     {f1_per_class[i]:.4f}     {roc_auc:.4f}     {pr_auc:.4f}\n"
        else:  # Non-driver class
            report += f"{class_name:>12}       {accuracy:.4f}      {precision_per_class[i]:.4f}     {recall_per_class[i]:.4f}     {f1_per_class[i]:.4f}         -         -\n"
    
    report += "\n"
    
    # Macro average
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    report += f"{'macro avg':>12}       {accuracy:.4f}      {macro_precision:.4f}     {macro_recall:.4f}     {macro_f1:.4f}     {roc_auc:.4f}     {pr_auc:.4f}\n"
    
    # Weighted average
    weighted_precision = np.average(precision_per_class, weights=support_per_class)
    weighted_recall = np.average(recall_per_class, weights=support_per_class)
    weighted_f1 = np.average(f1_per_class, weights=support_per_class)
    report += f"{'weighted avg':>12}       {accuracy:.4f}      {weighted_precision:.4f}     {weighted_recall:.4f}     {weighted_f1:.4f}     {roc_auc:.4f}     {pr_auc:.4f}\n"
    
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


def save_results_to_file(model_name, results, results_dir="./results"):
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


def summarize_default_results(results):
    """
    Cria um resumo dos resultados dos modelos padrão e salva em arquivo
    
    Args:
        results (list): Lista com resultados de todos os modelos
    """
    # Criar conteúdo do resumo
    content_lines = []
    content_lines.append("="*80)
    content_lines.append("RESUMO DOS RESULTADOS (PARÂMETROS PADRÃO)")
    content_lines.append("="*80)
    
    successful_models = [r for r in results if r['status'] == 'success']
    failed_models = [r for r in results if r['status'] == 'error']
    
    content_lines.append(f"Modelos executados com sucesso: {len(successful_models)}")
    content_lines.append(f"Modelos com erro: {len(failed_models)}")
    content_lines.append("")
    
    if successful_models:
        content_lines.append("COMPARAÇÃO DE PERFORMANCE (CONJUNTO DE TESTE):")
        content_lines.append("-" * 80)
        content_lines.append(f"{'Modelo':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'ROC AUC':<9} {'PR AUC':<8}")
        content_lines.append("-" * 80)
        
        for result in successful_models:
            metrics = result['test_metrics']
            content_lines.append(f"{result['model_name']:<25} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<11.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['f1_score']:<8.4f} "
                  f"{metrics['roc_auc']:<9.4f} "
                  f"{metrics['pr_auc']:<8.4f}")
    
    if failed_models:
        content_lines.append("")
        content_lines.append("MODELOS COM ERRO:")
        for result in failed_models:
            content_lines.append(f"  • {result['model_name']}: {result['error']}")
    
    content_lines.append("="*80)
    
    # Imprimir no terminal
    print("\n" + "\n".join(content_lines))
    
    # Salvar em arquivo
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_default_models_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(content_lines))
        f.write("\n")
    
    print(f"\nResumo salvo em: {filepath}")


def summarize_optimized_results(results):
    """
    Cria um resumo dos resultados dos modelos otimizados e salva em arquivo
    
    Args:
        results (list or dict): Lista com resultados dos modelos ou resultado único
    """
    # Criar conteúdo do resumo
    content_lines = []
    content_lines.append("="*80)
    content_lines.append("RESUMO DOS RESULTADOS (MODELOS OTIMIZADOS)")
    content_lines.append("="*80)
    
    # Se results for um único resultado (dict), converte para lista
    if isinstance(results, dict):
        results = [results]
    
    successful_models = [r for r in results if r['status'] == 'success']
    failed_models = [r for r in results if r['status'] == 'error']
    
    content_lines.append(f"Modelos executados com sucesso: {len(successful_models)}")
    content_lines.append(f"Modelos com erro: {len(failed_models)}")
    content_lines.append("")
    
    # Tabela de performance para todos os modelos (incluindo falhas)
    if successful_models or failed_models:
        content_lines.append("COMPARAÇÃO DE PERFORMANCE (CONJUNTO DE TESTE):")
        content_lines.append("-" * 80)
        content_lines.append(f"{'Modelo':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'ROC AUC':<9} {'PR AUC':<8}")
        content_lines.append("-" * 80)
        
        # Adicionar modelos bem-sucedidos
        for result in successful_models:
            # Verificar se o resultado tem test_metrics (modelos otimizados)
            if 'test_metrics' in result:
                metrics = result['test_metrics']
                # Para modelos otimizados, as métricas podem ter nomes ligeiramente diferentes
                f1_key = 'f1' if 'f1' in metrics else 'f1_score'
                content_lines.append(f"{result['model_name']:<25} "
                      f"{metrics['accuracy']:<10.4f} "
                      f"{metrics['precision']:<11.4f} "
                      f"{metrics['recall']:<8.4f} "
                      f"{metrics[f1_key]:<8.4f} "
                      f"{metrics['roc_auc']:<9.4f} "
                      f"{metrics['pr_auc']:<8.4f}")
        
        # Adicionar modelos com falha
        for result in failed_models:
            content_lines.append(f"{result['model_name']:<25} "
                  f"{'FAILED':<10} "
                  f"{'FAILED':<11} "
                  f"{'FAILED':<8} "
                  f"{'FAILED':<8} "
                  f"{'FAILED':<9} "
                  f"{'FAILED':<8}")
    
    if failed_models:
        content_lines.append("")
        content_lines.append("MODELOS COM ERRO:")
        for result in failed_models:
            content_lines.append(f"  • {result['model_name']}: {result['error']}")
    
    content_lines.append("="*80)
    
    # Imprimir no terminal
    for line in content_lines:
        print(line)
    
    # Salvar em arquivo
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_tuned_models_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(content_lines))
        f.write("\n")
    
    print(f"\nResumo salvo em: {filepath}")
