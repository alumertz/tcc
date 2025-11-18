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

def generate_experiment_folder_name(data_source="ana", mode="default", classification_type="binary", balance_strategy="none"):
    """
    Gera nome da pasta do experimento baseado na data e configurações
    Usa o timestamp global definido no início da execução
    
    Args:
        data_source (str): "ana" ou "renan"
        mode (str): "default" ou "optimized"  
        classification_type (str): "binary" ou "multiclass"
        balance_strategy (str): Estratégia de balanceamento utilizada
        
    Returns:
        str: Nome da pasta no formato "YYYYMMDD_HHMMSS_ana_default_binary_smoten"
    """
    timestamp = get_experiment_timestamp()
    
    # Normalizar valores para garantir consistência
    data_source = data_source.lower()
    mode = mode.lower()
    classification_type = classification_type.lower()
    balance_strategy = balance_strategy.lower()
    
    # Incluir estratégia de balanceamento apenas se não for "none"
    if balance_strategy == "none":
        folder_name = f"{timestamp}_{data_source}_{mode}_{classification_type}"
    else:
        folder_name = f"{timestamp}_{data_source}_{mode}_{classification_type}_{balance_strategy}"
    
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
                             classification_type="binary", balance_strategy="none", results_dir="./results"):
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
    experiment_folder = generate_experiment_folder_name(data_source, mode, classification_type, balance_strategy)
    
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
    
    # Salvar trials do Optuna em JSON (cleaned format)
    trials_list = results_data.get('trials', [])
    cleaned_trials = []
    for t in trials_list:
        try:
            cleaned = {
                'trial_number': int(t.get('trial_number', t.get('number', -1))),
                'params': t.get('params', {}),
                'score': float(t.get('score')) if t.get('score') is not None else None,
                'time': float(t.get('time')) if t.get('time') is not None else None
            }
            # Include per-fold CV metrics if present
            if isinstance(t, dict) and 'cv_metrics' in t and t['cv_metrics'] is not None:
                cleaned['cv_metrics'] = t['cv_metrics']
            cleaned_trials.append(cleaned)
        except Exception:
            # Fallback: include raw trial entry
            cleaned_trials.append(t)

    trials_file = os.path.join(model_dir, f"trials_{timestamp}.json")
    with open(trials_file, 'w') as f:
        json.dump(cleaned_trials, f, indent=2, ensure_ascii=False)
    
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


def save_detailed_results_txt(model_name, trials, test_metrics, best_params, optimization_info, output_path, param_importances_per_fold=None, aggregated_importances=None):
    """
    Salva um arquivo results.txt detalhado com todas as métricas de CV por trial, ordenado por PR AUC, para qualquer modelo.
    Também imprime importâncias dos hiperparâmetros por fold e agregadas.
    Args:
        model_name (str): Nome do modelo
        trials (list): Lista de dicts com dados dos trials (must include per-fold train/val metrics)
        test_metrics (dict): Métricas finais de teste
        best_params (dict): Melhores hiperparâmetros
        optimization_info (dict): Info sobre otimização (n_trials, tempo, etc)
        output_path (str): Caminho do arquivo para salvar
        param_importances_per_fold (list): Lista de dicts de importâncias por fold
        aggregated_importances (dict): Importâncias agregadas (média por parâmetro)
    """
    # Sort trials by PR AUC (descending)
    sorted_trials = sorted(trials, key=lambda t: t.get('score', 0), reverse=True)
    with open(output_path, 'w') as f:
        f.write(f"RESULTADOS DO MODELO: {model_name.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("MÉTRICAS DETALHADAS DE VALIDAÇÃO CRUZADA - TODOS OS TRIALS:\n" + "="*80 + "\n\n")
        # Print hyperparameter importances per fold
        if param_importances_per_fold:
            f.write("IMPORTÂNCIA DOS HIPERPARÂMETROS POR FOLD:\n" + "-"*50 + "\n")
            for i, imp in enumerate(param_importances_per_fold, 1):
                f.write(f"Fold {i}:\n")
                for k, v in imp.items():
                    f.write(f"  {k}: {v:.4f}\n")
                f.write("\n")
        # Print aggregated importances
        if aggregated_importances:
            f.write("IMPORTÂNCIA AGREGADA DOS HIPERPARÂMETROS (média dos folds):\n" + "-"*50 + "\n")
            for k, v in aggregated_importances.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
        for rank, trial in enumerate(sorted_trials, 1):
            pr_auc = trial.get('score', 0)
            f.write(f"TRIAL {trial.get('trial_number', rank)} (Rank #{rank}) - PR AUC: {pr_auc:.4f}\n")
            f.write("-"*60 + "\n")
            f.write("Hiperparâmetros:\n")
            for k, v in trial.get('params', {}).items():
                f.write(f"  {k}: {v}\n")
            f.write("\nMétricas por fold:\n")
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
            for metric in metrics:
                train_scores = trial.get(f'train_{metric}', [None]*5)
                test_scores = trial.get(f'test_{metric}', [None]*5)
                f.write(f"{metric.capitalize():<12} | ")
                for i in range(len(train_scores)):
                    f.write(f"Train: {train_scores[i]:.4f}  Test: {test_scores[i]:.4f} | ")
                f.write("\n")
            f.write("\n")
        f.write("AVALIAÇÃO NO CONJUNTO DE TESTE FINAL:\n" + "-"*50 + "\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nMELHORES HIPERPARÂMETROS:\n" + "-"*30 + "\n")
        f.write(json.dumps(best_params, indent=2, ensure_ascii=False) + "\n\n")
        # Destacar o melhor conjunto global
        final_best_params = optimization_info.get('final_best_params')
        final_best_score = optimization_info.get('final_best_score')
        if final_best_params:
            f.write("MELHOR CONJUNTO DE HIPERPARÂMETROS GLOBAL:\n" + "-"*60 + "\n")
            f.write(json.dumps(final_best_params, indent=2, ensure_ascii=False) + "\n")
            if final_best_score is not None:
                f.write(f"Score de interesse: {final_best_score:.4f}\n")
            f.write("\n")
        f.write("HISTÓRICO DE OTIMIZAÇÃO:\n" + "-"*30 + "\n")
        f.write(f"Número de trials: {optimization_info.get('n_trials', len(trials))}\n")
        f.write(f"Melhor score (CV): {optimization_info.get('best_score', sorted_trials[0].get('score', 0))}\n")
        f.write(f"Tempo total de otimização: {optimization_info.get('total_time', 0):.2f} segundos\n")


def save_detailed_results_txt_by_fold(model_name, all_folds_trials, output_path=None, final_best_params=None):
    
    with open(output_path, 'w') as f:
        f.write(f"RESULTADOS DO MODELO: {model_name.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for fold_data in all_folds_trials:
            fold = fold_data['fold']
            f.write(f"FOLD {fold} (loop externo):\n")
            f.write("-"*60 + "\n")
            f.write("Loop interno - Trials de hiperparâmetros testados:\n")
            for trial in fold_data['trials']:
                score = trial.get('score', 0)
                f.write(f"  Trial {trial['trial_number']}: Score={score:.4f} | Hiperparâmetros: {json.dumps(trial['params'])}\n")
            f.write("\nConjunto de hiperparâmetros escolhido:\n")
            for k, v in fold_data['best_params'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\nResultado do treino:\n")
            for k, v in fold_data['train_metrics'].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\nResultado do teste:\n")
            for k, v in fold_data['test_metrics'].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n" + "="*60 + "\n\n")
        # Tabela agregada de métricas
        f.write("TABELA DE MÉTRICAS DE TREINO E TESTE POR FOLD:\n")
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        header = "Fold".ljust(6) + "| " + " | ".join([m.ljust(10) for m in metrics])
        f.write(header + "\n" + "-"*len(header) + "\n")
        train_vals = {m: [] for m in metrics}
        test_vals = {m: [] for m in metrics}
        for fold_data in all_folds_trials:
            fold = str(fold_data['fold']).ljust(6)
            train_line = fold + "| " + " | ".join([f"{fold_data['train_metrics'][m]:.4f}".ljust(10) for m in metrics])
            test_line = fold + "| " + " | ".join([f"{fold_data['test_metrics'][m]:.4f}".ljust(10) for m in metrics])
            f.write("Treino: " + train_line + "\n")
            f.write("Teste:  " + test_line + "\n")
            for m in metrics:
                train_vals[m].append(fold_data['train_metrics'][m])
                test_vals[m].append(fold_data['test_metrics'][m])
        f.write("\n")
        # Tabela agregada de médias e desvios padrão
        f.write("MÉDIAS E DESVIOS PADRÃO DAS MÉTRICAS POR FOLD:\n")
        f.write(header + "\n" + "-"*len(header) + "\n")
        train_agg = "Treinos: ".ljust(9)
        test_agg = "Testes:  ".ljust(9)
        for m in metrics:
            mean_train = np.mean(train_vals[m])
            std_train = np.std(train_vals[m])
            mean_test = np.mean(test_vals[m])
            std_test = np.std(test_vals[m])
            train_agg += f"{mean_train:.4f}+-{std_train:.2f}".ljust(13)
            test_agg += f"{mean_test:.4f}+-{std_test:.2f}".ljust(13)
        f.write(train_agg + "\n")
        f.write(test_agg + "\n\n")
        # Melhor conjunto global
        if final_best_params:
            f.write("MELHOR CONJUNTO DE HIPERPARÂMETROS GLOBAL (selecionado pelo maior PR AUC  ):\n" + "-"*60 + "\n")
            f.write(json.dumps(final_best_params, indent=2, ensure_ascii=False) + "\n")
            f.write("\n")
        
