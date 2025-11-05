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


# Global variable to store the experiment timestamp for the session
_experiment_timestamp = None

def set_experiment_timestamp():
    """
    Define o timestamp do experimento para toda a sessão
    Deve ser chamado no início do main.py
    """
    global _experiment_timestamp
    if _experiment_timestamp is None:
        _experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _experiment_timestamp

def get_experiment_timestamp():
    """
    Retorna o timestamp do experimento atual
    Se não foi definido, cria um novo
    """
    global _experiment_timestamp
    if _experiment_timestamp is None:
        _experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _experiment_timestamp

def generate_experiment_folder_name(data_source="ana", mode="default", classification_type="binary"):
    """
    Gera nome da pasta do experimento baseado na data e configurações
    Usa o timestamp global definido no início da execução
    
    Args:
        data_source (str): "ana" ou "renan"
        mode (str): "default" ou "optimized"  
        classification_type (str): "binary" ou "multiclass"
        
    Returns:
        str: Nome da pasta no formato "YYYYMMDD_HHMMSS_ana_default_binary"
    """
    timestamp = get_experiment_timestamp()
    
    # Normalizar valores para garantir consistência
    data_source = data_source.lower()
    mode = mode.lower()
    classification_type = classification_type.lower()
    
    folder_name = f"{timestamp}_{data_source}_{mode}_{classification_type}"
    
    return folder_name


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

    # Prepare class names from unique labels
    classes = np.unique(y_true)
    class_names = [f'Class {c}' for c in classes]

    # Determine whether we have multiclass probabilities (2D) or binary (1D)
    is_multiclass_proba = hasattr(y_pred_proba, 'ndim') and getattr(y_pred_proba, 'ndim') == 2

    # Compute ROC AUC and PR AUC appropriately
    try:
        if is_multiclass_proba:
            # multiclass: compute both macro and weighted ROC AUC and per-class PR/ROC
            roc_auc_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            roc_auc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')

            pr_list = []
            roc_list = []
            support = []
            for i in range(y_pred_proba.shape[1]):
                yi = (y_true == classes[i]).astype(int)
                pr = average_precision_score(yi, y_pred_proba[:, i])
                # per-class ROC AUC (binarized)
                try:
                    roc_i = roc_auc_score(yi, y_pred_proba[:, i])
                except Exception:
                    roc_i = float('nan')
                pr_list.append(pr)
                roc_list.append(roc_i)
                support.append(int((y_true == classes[i]).sum()))

            # Weighted and macro PR AUC
            pr_auc_weighted = float(np.average(pr_list, weights=np.array(support)))
            pr_auc_macro = float(np.mean(pr_list))
        else:
            roc_auc_weighted = roc_auc_score(y_true, y_pred_proba)
            roc_auc_macro = roc_auc_weighted
            pr_auc_weighted = average_precision_score(y_true, y_pred_proba)
            pr_auc_macro = pr_auc_weighted
    except Exception:
        roc_auc_weighted = float('nan')
        roc_auc_macro = float('nan')
        pr_auc_weighted = float('nan')
        pr_auc_macro = float('nan')

    # Total support for summary rows
    total_support = np.sum(support_per_class)

    # Cabeçalho da tabela
    report = "              accuracy   precision    recall    f1-score   roc_auc    pr_auc\n\n"

    # Métricas por classe
    for i in range(len(precision_per_class)):
        class_name = class_names[i] if i < len(class_names) else f'Class {i}'
        # Per-class PR/AUC if available for multiclass
        if is_multiclass_proba:
            per_roc = roc_list[i] if i < len(roc_list) else float('nan')
            per_pr = pr_list[i] if i < len(pr_list) else float('nan')
            report += f"{class_name:>12}       {accuracy:.4f}      {precision_per_class[i]:.4f}     {recall_per_class[i]:.4f}     {f1_per_class[i]:.4f}     {per_roc:.4f}     {per_pr:.4f}\n"
        else:
            # Binary case prints pr/roc only for positive class (maintain similar layout)
            if i == 1:
                report += f"{class_name:>12}       {accuracy:.4f}      {precision_per_class[i]:.4f}     {recall_per_class[i]:.4f}     {f1_per_class[i]:.4f}     {roc_auc_weighted:.4f}     {pr_auc_weighted:.4f}\n"
            else:
                report += f"{class_name:>12}       {accuracy:.4f}      {precision_per_class[i]:.4f}     {recall_per_class[i]:.4f}     {f1_per_class[i]:.4f}         -         -\n"

    report += "\n"

    # Macro average
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    report += f"{'macro avg':>12}       {accuracy:.4f}      {macro_precision:.4f}     {macro_recall:.4f}     {macro_f1:.4f}     {roc_auc_macro if not np.isnan(roc_auc_macro) else '-'}     {pr_auc_macro if not np.isnan(pr_auc_macro) else '-'}\n"

    # Weighted average
    weighted_precision = np.average(precision_per_class, weights=support_per_class)
    weighted_recall = np.average(recall_per_class, weights=support_per_class)
    weighted_f1 = np.average(f1_per_class, weights=support_per_class)
    report += f"{'weighted avg':>12}       {accuracy:.4f}      {weighted_precision:.4f}     {weighted_recall:.4f}     {weighted_f1:.4f}     {roc_auc_weighted if not np.isnan(roc_auc_weighted) else '-'}     {pr_auc_weighted if not np.isnan(pr_auc_weighted) else '-'}\n"

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
            # Use string concatenation instead of f-string to avoid formatting issues
            file_handle.write("  " + str(param) + ": " + str(value) + "\n")
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
    file_handle.write("Validação cruzada com " + str(len(cv_metrics)) + " folds\n")


def save_model_results_unified(model_name, results_data, mode="default", data_source="ana", 
                             classification_type="binary", results_dir="./results"):
    """
    Função unificada para salvar resultados de modelos (padrão ou otimizados)
    
    Args:
        model_name (str): Nome do modelo
        results_data (dict): Dados dos resultados
        mode (str): "default" ou "optimized"
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
        results_dir (str): Diretório base para salvar
        
    Returns:
        tuple: Caminhos dos arquivos salvos
    """
    # Gerar nome da pasta do experimento
    experiment_folder = generate_experiment_folder_name(data_source, mode, classification_type)
    
    # Criar estrutura: results/YYYYMMDD_HHMMSS_ana_default_binary/model_name/
    experiment_dir = os.path.join(results_dir, experiment_folder)
    model_dir_name = model_name.lower().replace(' ', '_')
    model_dir = os.path.join(experiment_dir, model_dir_name)
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if mode == "default":
        return _save_default_mode_results(model_name, results_data, model_dir, timestamp)
    else:  # optimized
        return _save_optimized_mode_results(model_name, results_data, model_dir, timestamp)


def _save_default_mode_results(model_name, results_data, model_dir, timestamp):
    """Salva resultados do modo padrão (parâmetros padrão)"""
    
    # Estrutura dos dados para modo padrão
    structured_data = {
        'model_name': model_name,
        'mode': 'default_parameters',
        'cv_results': results_data.get('cv_results', {}),
        'test_metrics': results_data.get('test_metrics', {}),
        'parameters': results_data.get('parameters', {}),
        'timestamp': timestamp,
        # Include test predictions for plotting
        'test_predictions': results_data.get('test_predictions', {})
    }
    
    # Salvar métricas em JSON (sem timestamp no nome do arquivo)
    metrics_file = os.path.join(model_dir, "default_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    # Salvar relatório em texto (sem timestamp no nome do arquivo)
    report_file = os.path.join(model_dir, "default_results.txt")
    with open(report_file, 'w') as f:
        f.write(f"MODELO: {model_name} (Parâmetros Padrão)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Validação cruzada
        f.write("VALIDAÇÃO CRUZADA (5-fold):\n")
        f.write("-"*50 + "\n")
        cv_results = results_data.get('cv_results', {})
        for metric, result in cv_results.items():
            if isinstance(result, dict) and 'mean' in result and 'std' in result:
                f.write(f"  {metric.upper()}: {result['mean']:.4f} ± {result['std']:.4f}\n")
        
        # Teste final
        f.write("\nTESTE FINAL:\n")
        f.write("-"*50 + "\n")
        test_metrics = results_data.get('test_metrics', {})
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"  {metric.upper()}: {value:.4f}\n")
        
        # Relatório de classificação
        if 'classification_report' in results_data:
            f.write(f"\nRELATÓRIO DE CLASSIFICAÇÃO:\n")
            f.write("-"*50 + "\n")
            f.write(results_data['classification_report'])
        
        # Parâmetros
        f.write(f"\nPARÂMETROS:\n")
        f.write("-"*50 + "\n")
        f.write(json.dumps(results_data.get('parameters', {}), indent=2))
        f.write("\n")
    
    print(f"Resultados padrão salvos em: {model_dir}")
    return metrics_file, report_file


def _save_optimized_mode_results(model_name, results_data, model_dir, timestamp):
    """Salva resultados do modo otimizado (com trials Optuna)"""
    
    # Salvar trials do Optuna em JSON
    trials_file = os.path.join(model_dir, f"trials_{timestamp}.json")
    with open(trials_file, 'w') as f:
        json.dump(results_data.get('trials', []), f, indent=2, default=str)
    
    # Salvar relatório de teste em texto
    test_results_file = os.path.join(model_dir, f"test_results_{timestamp}.txt")
    with open(test_results_file, 'w') as f:
        f.write(f"RESULTADOS DO MODELO: {model_name.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Métricas de CV detalhadas se disponível
        if 'all_cv_metrics' in results_data:
            generate_all_trials_cv_tables(f, results_data['all_cv_metrics'])
        elif 'cv_detailed_metrics' in results_data:
            generate_cv_metrics_table(f, results_data['cv_detailed_metrics'])
        
        # Avaliação final no teste
        f.write("AVALIAÇÃO NO CONJUNTO DE TESTE FINAL:\n")
        f.write("-"*50 + "\n")
        test_metrics = results_data.get('test_metrics', {})
        
        # Compatibilidade com diferentes chaves de F1
        f1_key = 'f1_score' if 'f1_score' in test_metrics else 'f1'
        
        f.write(f"Acurácia: {test_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Precisão: {test_metrics.get('precision', 0):.4f}\n")
        f.write(f"Recall: {test_metrics.get('recall', 0):.4f}\n")
        f.write(f"F1-Score: {test_metrics.get(f1_key, 0):.4f}\n")
        f.write(f"ROC AUC: {test_metrics.get('roc_auc', 0):.4f}\n")
        f.write(f"PR AUC: {test_metrics.get('pr_auc', 0):.4f}\n\n")
        
        # Relatório de classificação se disponível
        if 'classification_report' in test_metrics:
            f.write("RELATÓRIO DETALHADO:\n")
            f.write("-"*30 + "\n")
            f.write(test_metrics['classification_report'])
            f.write("\n\n")
        
        # Melhores hiperparâmetros
        f.write("MELHORES HIPERPARÂMETROS:\n")
        f.write("-"*30 + "\n")
        f.write(json.dumps(results_data.get('best_params', {}), indent=2))
        f.write("\n\n")
        
        # Histórico de otimização
        f.write("HISTÓRICO DE OTIMIZAÇÃO:\n")
        f.write("-"*30 + "\n")
        f.write(f"Número de trials: {len(results_data.get('trials', []))}\n")
        f.write(f"Melhor score (CV): {results_data.get('best_score', 0):.4f}\n")
        f.write(f"Tempo total de otimização: {results_data.get('optimization_time', 0):.2f} segundos\n")
    
    print(f"Resultados otimizados salvos em: {model_dir}")
    return trials_file, test_results_file


def summarize_results(results, mode="default", data_source="ana", classification_type="binary"):
    """
    Cria um resumo unificado dos resultados dos modelos e salva em arquivo
    
    Args:
        results (list or dict): Lista com resultados dos modelos ou resultado único
        mode (str): Modo de execução - "default" ou "optimized"
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
    """
    # Configurações baseadas no modo
    if mode == "default":
        title = "RESUMO DOS RESULTADOS (PARÂMETROS PADRÃO)"
        filename_prefix = "summary_default_models"
    else:  # optimized
        title = "RESUMO DOS RESULTADOS (MODELOS OTIMIZADOS)"
        filename_prefix = "summary_tuned_models"
    
    # Se results for um único resultado (dict), converte para lista
    if isinstance(results, dict):
        results = [results]
    
    # Filtrar modelos por status
    successful_models = [r for r in results if r.get('status') == 'success']
    failed_models = [r for r in results if r.get('status') == 'error']
    
    # Criar conteúdo do resumo
    content_lines = []
    content_lines.append("="*80)
    content_lines.append(title)
    content_lines.append("="*80)
    
    content_lines.append(f"Modelos executados com sucesso: {len(successful_models)}")
    content_lines.append(f"Modelos com erro: {len(failed_models)}")
    content_lines.append("")
    
    # Tabela de performance dos modelos bem-sucedidos
    if successful_models:
        # Coletar dados dos modelos para ordenação
        models_data = []
        for result in successful_models:
            # Handle both nested CV results and regular test results
            if 'test_metrics' in result:
                # Regular optimization with test metrics
                metrics = result['test_metrics']
                # Compatibilidade com diferentes chaves F1
                f1_key = 'f1_score' if 'f1_score' in metrics else 'f1'
                
                models_data.append({
                    'name': result['model_name'],
                    'metrics': metrics,
                    'f1_key': f1_key,
                    'result_type': 'test_metrics'
                })
            elif hasattr(result, 'keys') and any('mean' in str(v) for v in result.values() if isinstance(v, dict)):
                # Nested CV results (aggregated metrics format)
                # Extract mean values from nested CV aggregated metrics
                nested_metrics = {
                    'accuracy': result.get('accuracy', {}).get('mean', 0.0),
                    'precision': result.get('precision', {}).get('mean', 0.0),
                    'recall': result.get('recall', {}).get('mean', 0.0),
                    'f1': result.get('f1', {}).get('mean', 0.0),
                    'roc_auc': result.get('roc_auc', {}).get('mean', 0.0),
                    'pr_auc': result.get('roc_auc', {}).get('mean', 0.0)  # Use ROC AUC as proxy for PR AUC in nested CV
                }
                
                models_data.append({
                    'name': result['model_name'],
                    'metrics': nested_metrics,
                    'f1_key': 'f1',
                    'result_type': 'nested_cv'
                })
        
        # Ordenar por PR AUC (métrica principal) em ordem decrescente
        # Handle cases where pr_auc might not be available
        models_data.sort(key=lambda x: x['metrics'].get('pr_auc', x['metrics'].get('roc_auc', 0.0)), reverse=True)
        
        # Cabeçalho da tabela
        content_lines.append("COMPARAÇÃO DE PERFORMANCE (CONJUNTO DE TESTE):")
        content_lines.append("-" * 90)
        content_lines.append(f"{'Rank':<5} {'Modelo':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'ROC AUC':<9} {'PR AUC':<8}")
        content_lines.append("-" * 90)
        
        # Adicionar modelos ordenados por ranking
        for rank, model_data in enumerate(models_data, 1):
            metrics = model_data['metrics']
            f1_key = model_data['f1_key']
            
            # Destacar o melhor modelo
            rank_display = f"🥇{rank}" if rank == 1 else f"  {rank}"
            
            # Handle missing metrics gracefully
            accuracy = metrics.get('accuracy', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            f1_score = metrics.get(f1_key, 0.0)
            roc_auc = metrics.get('roc_auc', 0.0)
            pr_auc = metrics.get('pr_auc', roc_auc)  # Use ROC AUC as fallback for PR AUC
            
            content_lines.append(f"{rank_display:<5} {model_data['name']:<25} "
                  f"{accuracy:<10.4f} "
                  f"{precision:<11.4f} "
                  f"{recall:<8.4f} "
                  f"{f1_score:<8.4f} "
                  f"{roc_auc:<9.4f} "
                  f"{pr_auc:<8.4f}")
        
        # Estatísticas do melhor modelo
        if models_data:
            best_model = models_data[0]
            best_metrics = best_model['metrics']
            content_lines.append("")
            content_lines.append("🏆 MELHOR MODELO:")
            content_lines.append(f"   Modelo: {best_model['name']}")
            content_lines.append(f"   PR AUC: {best_metrics['pr_auc']:.4f}")
            content_lines.append(f"   ROC AUC: {best_metrics['roc_auc']:.4f}")
            content_lines.append(f"   F1-Score: {best_metrics[best_model['f1_key']]:.4f}")
            content_lines.append(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    # Adicionar modelos com falha
    if failed_models:
        if successful_models:
            # Adicionar falhas na tabela
            for result in failed_models:
                content_lines.append(f"{'  X':<5} {result['model_name']:<25} "
                      f"{'FAILED':<10} "
                      f"{'FAILED':<11} "
                      f"{'FAILED':<8} "
                      f"{'FAILED':<8} "
                      f"{'FAILED':<9} "
                      f"{'FAILED':<8}")
        
        content_lines.append("")
        content_lines.append("❌ MODELOS COM ERRO:")
        for result in failed_models:
            content_lines.append(f"   • {result['model_name']}: {result.get('error', 'Erro não especificado')}")
    
    content_lines.append("="*80)
    
    # Imprimir no terminal
    if mode == "default":
        print("\n" + "\n".join(content_lines))
    else:
        for line in content_lines:
            print(line)
    
    # Salvar em arquivo na pasta do experimento
    experiment_folder = generate_experiment_folder_name(data_source, mode, classification_type)
    experiment_dir = os.path.join("./results", experiment_folder)
    os.makedirs(experiment_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.txt"
    filepath = os.path.join(experiment_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Configuração: {data_source.upper()} + {mode.upper()} + {classification_type.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write("\n".join(content_lines))
        f.write("\n")
    
    print(f"\nResumo salvo em: {filepath}")


def save_nested_cv_results(model_name, aggregated_metrics, best_params_per_fold, 
                          data_source="ana", classification_type="binary", 
                          n_trials=100, outer_cv_folds=5):
    """
    Salva resultados de nested cross-validation em formato JSON
    
    Args:
        model_name (str): Nome do modelo
        aggregated_metrics (dict): Métricas agregadas com mean, std e scores
        best_params_per_fold (list): Lista dos melhores parâmetros por fold
        data_source (str): Fonte dos dados
        classification_type (str): Tipo de classificação
        n_trials (int): Número de trials utilizados
        outer_cv_folds (int): Número de folds externos
    """
    # Criar estrutura de resultados compatível
    nested_cv_results = {
        'model_name': model_name,
        'optimization_type': 'nested_cross_validation',
        'configuration': {
            'outer_cv_folds': outer_cv_folds,
            'n_trials_per_fold': n_trials,
            'data_source': data_source,
            'classification_type': classification_type,
            'timestamp': datetime.now().isoformat()
        },
        'aggregated_metrics': aggregated_metrics,
        'best_params_per_fold': best_params_per_fold,
        'nested_cv_summary': {
            'mean_accuracy': aggregated_metrics['accuracy']['mean'],
            'std_accuracy': aggregated_metrics['accuracy']['std'],
            'mean_f1': aggregated_metrics['f1']['mean'],
            'std_f1': aggregated_metrics['f1']['std'],
            'mean_roc_auc': aggregated_metrics['roc_auc']['mean'],
            'std_roc_auc': aggregated_metrics['roc_auc']['std']
        }
    }
    
    # Criar diretório do experimento
    experiment_folder = generate_experiment_folder_name(data_source, "optimized", classification_type)
    experiment_dir = os.path.join("./results", experiment_folder)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nested_cv_{model_name}_{data_source}_{classification_type}_{timestamp}.json"
    filepath = os.path.join(experiment_dir, filename)
    
    # Converter numpy arrays para listas para serialização JSON
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Converter dados para formato serializável
    serializable_results = convert_numpy_types(nested_cv_results)
    
    # Salvar arquivo JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados de Nested CV salvos em: {filepath}")
