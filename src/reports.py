import json
import numpy as np
from datetime import datetime
import os

# Global variable to store the experiment timestamp for the session
_experiment_timestamp = None

"""
Módulo para geração de relatórios e salvamento de resultados de modelos de machine learning.
Contém funções para formatação de resultados de validação cruzada, relatórios de classificação e persistência.
"""

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

def save_optimization_figs(fold_results, model_dir):
    """
    Save Optuna optimization figures for each fold to the model directory.
    Each figure is saved as a PNG file with a name indicating the fold and figure type.
    """
    import os
    for fr in fold_results:
        fold_id = fr.fold
        # Save optimization history figure
        if fr.opt_hist_fig is not None:
            hist_path = os.path.join(model_dir, f"fold_{fold_id}_opt_history.png")
            try:
                fr.opt_hist_fig.write_image(hist_path)
            except Exception as e:
                print(f"Erro ao salvar opt_history para fold {fold_id}: {e}")
        # Save contour figure
        if fr.plot_contour_fig is not None:
            contour_path = os.path.join(model_dir, f"fold_{fold_id}_opt_contour.png")
            try:
                fr.plot_contour_fig.write_image(contour_path)
            except Exception as e:
                print(f"Erro ao salvar opt_contour para fold {fold_id}: {e}")

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
            f.write(f"\nMelhor trial (Optuna): {fold_data['best_trial_number']}\n")
            for k, v in fold_data['best_params'].items():
                f.write(f"  {k}: {v}\n")
            # Print parameter importances if available
            f.write("\nImportância dos parâmetros (Optuna):\n")
            for param, importance in fold_data['param_importances'].items():
                f.write(f"  {param}: {importance:.4f}\n")
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
        try: 
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
        except KeyError as e:
            f.write(f"Erro ao gerar tabela de métricas: chave ausente {e}\n\n")
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
        

def save_holdout_results(model_name, holdout_results, data_source, classification_type):
    """Save holdout evaluation results to files"""
    
    
    # Get the same experiment directory as the optimization results
    experiment_folder = generate_experiment_folder_name(data_source, "optimized", classification_type)
    experiment_dir = os.path.join("./results", experiment_folder)
    model_dir_name = model_name.lower().replace(' ', '_')
    model_dir = os.path.join(experiment_dir, model_dir_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save holdout results
    holdout_results_path = os.path.join(model_dir, "holdout_results.txt")
    with open(holdout_results_path, 'w') as f:
        f.write(f"Holdout Evaluation Results for {model_name}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Results for each fold's best parameters on holdout set:\n")
        f.write("-"*50 + "\n")
        
        for hr in holdout_results:
            f.write(f"\nFold {hr['fold']}:\n")
            f.write(f"  Best Parameters: {hr['best_params']}\n")
            f.write(f"  Holdout Metrics:\n")
            f.write(f"    Accuracy: {hr['holdout_metrics'].accuracy:.4f}\n")
            f.write(f"    Precision: {hr['holdout_metrics'].precision:.4f}\n")
            f.write(f"    Recall: {hr['holdout_metrics'].recall:.4f}\n")
            f.write(f"    F1-Score: {hr['holdout_metrics'].f1:.4f}\n")
            f.write(f"    ROC-AUC: {hr['holdout_metrics'].roc_auc:.4f}\n")
            f.write(f"    PR-AUC: {hr['holdout_metrics'].pr_auc:.4f}\n")
        
        # Find and write best performing model
        best_holdout_idx = np.argmax([hr['holdout_metrics'].pr_auc for hr in holdout_results])
        best_holdout = holdout_results[best_holdout_idx]
        
        f.write(f"\n" + "="*50 + "\n")
        f.write(f"BEST PERFORMING MODEL ON HOLDOUT SET:\n")
        f.write(f"Fold {best_holdout['fold']} (PR-AUC: {best_holdout['holdout_metrics'].pr_auc:.4f})\n")
        f.write(f"Parameters: {best_holdout['best_params']}\n")
        f.write(f"Final Holdout Performance:\n")
        f.write(f"  Accuracy: {best_holdout['holdout_metrics'].accuracy:.4f}\n")
        f.write(f"  Precision: {best_holdout['holdout_metrics'].precision:.4f}\n")
        f.write(f"  Recall: {best_holdout['holdout_metrics'].recall:.4f}\n")
        f.write(f"  F1-Score: {best_holdout['holdout_metrics'].f1:.4f}\n")
        f.write(f"  ROC-AUC: {best_holdout['holdout_metrics'].roc_auc:.4f}\n")
        f.write(f"  PR-AUC: {best_holdout['holdout_metrics'].pr_auc:.4f}\n")
    
    print(f"Holdout evaluation results saved to: {holdout_results_path}")


def default_report(model_name, folds_metrics, test_metrics, output_path=None, balance_strategy="none"):
    """
    Gera um relatório simples para avaliação padrão de um modelo.
    Args:
        model_name (str): Nome do modelo
        folds_metrics (list): Lista de dicts com métricas de treino e validação por fold
        test_metrics (dict): Métricas finais de teste
        output_path (str, opcional): Caminho para salvar o relatório. Se None, só retorna string.
    Returns:
        str: Relatório gerado
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    extra_metrics = [
        'per_class_roc_auc', 'per_class_pr_auc',
        'pr_auc_macro', 'pr_auc_weighted', 'pr_auc_micro',
        'roc_auc_macro', 'roc_auc_weighted', 'roc_auc_micro'
    ]
    report = f"RELATÓRIO PADRÃO DO MODELO: {model_name} com {balance_strategy}\n" + "="*80 + "\n\n"
    report += "MÉTRICAS DE TREINO (HOLDOUT):\n"
    if folds_metrics and isinstance(folds_metrics, dict):
        train_metrics = folds_metrics.get('train_metrics', {})
        for m in metrics:
            v = train_metrics.get(m, None)
            if v is not None:
                report += f"{m.capitalize()}: {v:.4f}\n"
        # Print extra metrics for multiclass
        for m in extra_metrics:
            v = train_metrics.get(m, None)
            if v is not None:
                if m.startswith('per_class_') and isinstance(v, dict):
                    report += f"{m.replace('_', ' ').capitalize()}:\n"
                    for cls, score in v.items():
                        report += f"  Classe {int(cls)}: {score:.4f}\n"
                else:
                    report += f"{m.replace('_', ' ').capitalize()}: {v:.4f}\n"
    elif folds_metrics and isinstance(folds_metrics, list) and len(folds_metrics) == 1:
        train_metrics = folds_metrics[0].get('train_metrics', {})
        for m in metrics:
            v = train_metrics.get(m, None)
            if v is not None:
                report += f"{m.capitalize()}: {v:.4f}\n"
        for m in extra_metrics:
            v = train_metrics.get(m, None)
            if v is not None:
                if m.startswith('per_class_') and isinstance(v, dict):
                    report += f"{m.replace('_', ' ').capitalize()}:\n"
                    for cls, score in v.items():
                        report += f"  Classe {int(cls)}: {score:.4f}\n"
                else:
                    report += f"{m.replace('_', ' ').capitalize()}: {v:.4f}\n"
    report += "\nMÉTRICAS DE TESTE (HOLDOUT):\n" + "-"*30 + "\n"
    for m in metrics:
        v = test_metrics.get(m, None)
        if v is not None:
            report += f"{m.capitalize()}: {v:.4f}\n"
    for m in extra_metrics:
        v = test_metrics.get(m, None)
        if v is not None:
            if m.startswith('per_class_') and isinstance(v, dict):
                report += f"{m.replace('_', ' ').capitalize()}:\n"
                for cls, score in v.items():
                    report += f"  Classe {int(cls)}: {score:.4f}\n"
            else:
                report += f"{m.replace('_', ' ').capitalize()}: {v:.4f}\n"
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

        # Save metrics and test_predictions in metrics.json for plotting compatibility
        # Only save if test_metrics contains y_true, y_pred, y_pred_proba
        metrics_json_path = os.path.join(os.path.dirname(output_path), 'metrics.json')
        test_predictions = None
        # Try to get y_true, y_pred, y_pred_proba from test_metrics
        y_true = test_metrics.get('y_true', None)
        y_pred = test_metrics.get('y_pred', None)
        y_pred_proba = test_metrics.get('y_pred_proba', None)
        # Convert to lists if they are numpy arrays
        import numpy as np
        if y_true is not None:
            y_true = y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
        if y_pred is not None:
            y_pred = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
        if y_pred_proba is not None:
            y_pred_proba = y_pred_proba.tolist() if hasattr(y_pred_proba, 'tolist') else list(y_pred_proba)
        if y_true is not None and y_pred is not None and y_pred_proba is not None:
            test_predictions = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        # Ensure all metrics are serializable
        def make_serializable(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj

        metrics_data = {
            'test_metrics': make_serializable(test_metrics),
            'train_metrics': make_serializable(folds_metrics.get('train_metrics', {}) if isinstance(folds_metrics, dict) else {}),
            'test_predictions': make_serializable(test_predictions)
        }
        import json
        with open(metrics_json_path, 'w') as mf:
            json.dump(metrics_data, mf, indent=2, ensure_ascii=False)
    return report

def save_default_experiment_summary(experiment_dir, results, balance_strategy=""):
    """
    Cria um arquivo resumo com uma tabela de métricas para todos os algoritmos executados com a estratégia de balanceamento escolhida.
    Args:
        experiment_dir (str): Pasta principal do experimento
        results (list): Lista de dicts com resultados de cada modelo
        balance_strategy (str): Estratégia de balanceamento
    """
    import os
    import numpy as np
    summary_path = os.path.join(experiment_dir, f"summary_{balance_strategy}.txt")
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    with open(summary_path, 'w') as f:
        f.write(f"RESUMO DOS RESULTADOS - Balanceamento: {balance_strategy}\n")
        f.write("="*80 + "\n\n")
        header = "Modelo".ljust(25) + "| " + " | ".join([m.ljust(10) for m in metrics])
        f.write(header + "\n" + "-"*len(header) + "\n")
        for result in results:
            if result is None:
                continue
            model_name = result.get('model_name', '-')
            test_metrics = result.get('test_metrics', {})
            line = model_name.ljust(25) + "| "
            for m in metrics:
                v = test_metrics.get(m, None)
                if v is not None:
                    line += f"{v:.4f}".ljust(10) + "| "
                else:
                    line += "-".ljust(10) + "| "
            f.write(line + "\n")
        f.write("\n")

def format_5fold_report(model_name, folds_results, aggregated, classification_type):
    lines = []
    lines.append(f"MODEL: {model_name}")
    lines.append("=" * 70)

    # ---------------------------------------------------------
    # PER FOLD
    # ---------------------------------------------------------
    for fold in folds_results:
        lines.append(f"\nFold {fold['fold']}")
        lines.append("-" * 30)

        # TRAIN
        lines.append(" TRAINING METRICS:")
        for k, v in fold["train"].items():
            if isinstance(v, (float, int)):
                lines.append(f"   {k}: {v}")

        # VALIDATION
        lines.append("\n VALIDATION METRICS:")
        for k, v in fold["val"].items():
            if isinstance(v, (float, int)):
                lines.append(f"   {k}: {v}")

        # MULTICLASS extras
        if classification_type == "multiclass":
            pc_roc = fold["val"].get("per_class_roc_auc", {})
            pc_pr  = fold["val"].get("per_class_pr_auc", {})

            lines.append("\n   Per class roc auc:")
            for cls, val in pc_roc.items():
                lines.append(f"     Classe {cls}: {val}")

            lines.append("   Per class pr auc:")
            for cls, val in pc_pr.items():
                lines.append(f"     Classe {cls}: {val}")

            lines.append(f"   Pr auc macro: {fold['val'].get('pr_auc_macro')}")
            lines.append(f"   Pr auc weighted: {fold['val'].get('pr_auc_weighted')}")
            lines.append(f"   Pr auc micro: {fold['val'].get('pr_auc_micro')}")

            lines.append(f"   Roc auc macro: {fold['val'].get('roc_auc_macro')}")
            lines.append(f"   Roc auc weighted: {fold['val'].get('roc_auc_weighted')}")
            lines.append(f"   Roc auc micro: {fold['val'].get('roc_auc_micro')}")

    # ---------------------------------------------------------
    # AGGREGATED RESULT
    # ---------------------------------------------------------
    lines.append("\nAGGREGATED VALIDATION METRICS (MEAN OF 5 FOLDS)")
    lines.append("=" * 70)

    for k, v in aggregated.items():
        lines.append(f"{k}: {v}")

    return "\n".join(lines)