#!/usr/bin/env python3
"""
Funções para criar gráficos de ROC Curve e Precision-Recall Curve
a partir dos resultados dos testes de algoritmos de classificação.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix
from plot_curves_multiclass import generate_multiclass_plots
import warnings
warnings.filterwarnings('ignore')

# Adicionar path do projeto
sys.path.append('/Users/i583975/git/tcc')

# Configuração do matplotlib para melhor qualidade
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Configuração do seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

# Paleta de cores padronizada para todos os modelos
STANDARD_COLORS = {
    'decisiontree': '#FF0000',           # Vermelho
    'randomforest': '#FF7F00',           # Laranja
    'gradientboosting': '#FFFF00',       # Amarelo
    'histogramgradientboosting': '#00FF00',  # Verde
    'knearestneighbors': '#0000FF',     # Azul
    'multilayerperceptron': '#4B0082',  # Índigo
    'supportvectorclassifier': '#9400D3',  # Violeta
    'catboost': '#FF1493',                # Rosa forte
    'xgboost': '#00CED1',                 # Turquesa
}

# Mapeamento de nomes de modelo para nomes de exibição
MODEL_DISPLAY_NAMES = {
    'catboost': 'CatBoost',
    'Catboost': 'CatBoost',
    'decision_tree': 'Decision Tree',
    'DT': 'Decision Tree',
    'dt': 'Decision Tree',
    'gradient_boosting': 'Gradient Boosting',
    'GB': 'Gradient Boosting',
    'gb': 'Gradient Boosting',
    'histogram_gradient_boosting': 'Histogram Gradient Boosting',
    'HGB': 'Histogram Gradient Boosting',
    'hgb': 'Histogram Gradient Boosting',
    'k_nearest_neighbors': 'K-Nearest Neighbors',
    'KNN': 'K-Nearest Neighbors',
    'Knn': 'K-Nearest Neighbors',
    'knn': 'K-Nearest Neighbors',
    'multi_layer_perceptron': 'Multi-Layer Perceptron',
    'MLP': 'Multi-Layer Perceptron',
    'Mlp': 'Multi-Layer Perceptron',
    'mlp': 'Multi-Layer Perceptron',
    'random_forest': 'Random Forest',
    'RF': 'Random Forest',
    'rf': 'Random Forest',
    'support_vector_classifier': 'Support Vector Classifier',
    'SVC': 'Support Vector Classifier',
    'Svc': 'Support Vector Classifier',
    'svc': 'Support Vector Classifier',
    'xgboost': 'XGBoost',
    'Xgboost': 'XGBoost',
}


def format_model_name(model_name):
    """
    Formata o nome do modelo para exibição, removendo prefixos como 'metrics' ou 'Metrics'
    
    Args:
        model_name (str): Nome do modelo (pode incluir prefixos)
        
    Returns:
        str: Nome formatado para exibição
    """
    # Remover prefixos comuns (case-insensitive)
    clean_name = model_name
    prefixes_to_remove = ['metrics_', 'metrics', 'Metrics']
    
    for prefix in prefixes_to_remove:
        if clean_name.lower().startswith(prefix.lower()):
            clean_name = clean_name[len(prefix):]
            break
    
    # Buscar no dicionário de nomes de exibição (exact match first)
    if clean_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[clean_name]
    
    # Se não encontrou no dicionário, formatar manualmente
    # Substituir underscores e hífens por espaços e capitalizar
    formatted = clean_name.replace('_', ' ').replace('-', ' ').title()
    
    return formatted


def create_plots_directory(experiment_dir):
    """Cria diretório curves dentro do diretório do experimento para salvar os gráficos"""
    curves_dir = os.path.join(experiment_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    
    return curves_dir


def load_from_forplots_directory(forplots_dir):
    """
    Carrega predições de um diretório forplots com arquivos metrics.json nomeados por modelo
    
    Args:
        forplots_dir (str): Diretório contendo arquivos metrics_[model_name].json
        
    Returns:
        dict: Dicionário com predições por modelo
    """
    models_data = {}
    
    if not os.path.exists(forplots_dir):
        print(f"❌ Diretório {forplots_dir} não existe")
        return models_data
    
    print(f"📂 Carregando arquivos de: {forplots_dir}")
    
    # Listar todos os arquivos .json no diretório
    json_files = [f for f in os.listdir(forplots_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"⚠️  Nenhum arquivo JSON encontrado em {forplots_dir}")
        return models_data
    
    print(f"Encontrados {len(json_files)} arquivos JSON")
    
    for json_file in json_files:
        file_path = os.path.join(forplots_dir, json_file)
        
        # Extrair nome do modelo do arquivo (remover extensão .json)
        model_name = json_file.replace('.json', '')
        
        # Se o arquivo começa com "metrics_", remover esse prefixo
        if model_name.startswith('metrics_'):
            model_name = model_name.replace('metrics_', '', 1)
        
        print(f"  Carregando {json_file} como '{model_name}'...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Verificar se tem as métricas esperadas no arquivo
            has_metrics = False
            stored_pr_auc = None
            stored_roc_auc = None
            
            # Tentar extrair métricas salvas do arquivo
            if 'test_metrics' in data:
                test_metrics = data['test_metrics']
                if 'average_precision' in test_metrics:
                    stored_pr_auc = test_metrics['average_precision']
                    has_metrics = True
                if 'roc_auc' in test_metrics:
                    stored_roc_auc = test_metrics['roc_auc']
                    has_metrics = True
            
            if has_metrics:
                print(f"    📊 Métricas encontradas no arquivo:")
                if stored_pr_auc is not None:
                    print(f"       PR AUC: {stored_pr_auc:.4f}")
                if stored_roc_auc is not None:
                    print(f"       ROC AUC: {stored_roc_auc:.4f}")
            
            if 'test_predictions' in data:
                predictions = data['test_predictions']
                if predictions and 'y_true' in predictions and 'y_pred_proba' in predictions:
                    models_data[model_name] = {
                        'predictions': predictions,
                        'file_path': file_path,
                        'stored_metrics': {
                            'pr_auc': stored_pr_auc,
                            'roc_auc': stored_roc_auc
                        }
                    }
                    print(f"    ✓ {len(predictions['y_true'])} amostras carregadas")
                else:
                    print(f"    ⚠️  test_predictions incompleto")
            else:
                print(f"    ⚠️  sem test_predictions")
        except Exception as e:
            print(f"    ❌ Erro ao carregar: {e}")
    
    return models_data


def load_saved_predictions(results_dir="./results", use_forplots=False, forplots_path=None):
    """
    Carrega predições salvas de todos os modelos testados (modo default ou optimized)
    
    Args:
        results_dir (str): Diretório para procurar resultados
        use_forplots (bool): Se True, usa o diretório forplots_path
        forplots_path (str): Caminho para o diretório forplots
        
    Returns:
        tuple: (models_data, experiment_dir, mode) - Dicionário com predições, caminho do experimento e modo (default/optimized)
    """
    models_data = {}
    experiment_dir = None
    detected_mode = None
    
    # Se use_forplots está ativo, carregar do diretório forplots
    if use_forplots and forplots_path:
        models_data = load_from_forplots_directory(forplots_path)
        experiment_dir = os.path.dirname(forplots_path)
        detected_mode = 'forplots'
        return models_data, experiment_dir, detected_mode
    
    # Lista de modelos padronizada (sem variações)
    model_names = [
        'decision_tree', 'random_forest', 'gradient_boosting', 
        'histogram_gradient_boosting', 'k_nearest_neighbors', 'multi_layer_perceptron', 
        'support_vector_classifier', 'catboost', 'svc'
    ]
    
    print(f"Procurando resultados em: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"Diretório {results_dir} não existe")
        return models_data, experiment_dir, detected_mode
    
    # Listar subdiretórios de experimentos (formato: YYYYMMDD_HHMMSS_ana_default_* ou _optimized_*)
    experiment_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Verificar se segue o padrão de timestamp de experimento
            if len(item) > 15 and item[8] == '_' and item[15] == '_' and ('binary' in item or 'multiclass' in item):
                experiment_dirs.append(item)
    
    if not experiment_dirs:
        print(f"Nenhum experimento encontrado em {results_dir}")
        print(f"Diretórios disponíveis: {os.listdir(results_dir)}")
        return models_data, experiment_dir, detected_mode
    
    # Ordenar diretórios por timestamp (mais recente primeiro)
    experiment_dirs.sort(reverse=True)
    
    # Mostrar opções para o usuário
    print("\nDiretórios de experimentos disponíveis:")
    for i, dir_name in enumerate(experiment_dirs, 1):
        # Extrair informações do nome do diretório para melhor visualização
        parts = dir_name.split('_')
        if len(parts) >= 10:
            date_part = parts[0]  # YYYYMMDD
            time_part = parts[1]  # HHMMSS
            mode_part = parts[3]  # default or optimized
            classification_type = parts[-1]  # binary/multiclass
            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
            print(f"{i}. {dir_name}")
            print(f"   Data: {formatted_date} {formatted_time} | Modo: {mode_part.upper()} | Tipo: {classification_type}")
        else:
            print(f"{i}. {dir_name}")
    
    # Solicitar seleção do usuário
    print(f"\n[Enter] = usar o mais recente ({experiment_dirs[0]})")
    choice = input("Selecione o número do experimento (ou pressione Enter): ").strip()
    
    if choice == "":
        selected_experiment = experiment_dirs[0]
        print(f"Usando experimento mais recente: {selected_experiment}")
    else:
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(experiment_dirs):
                selected_experiment = experiment_dirs[choice_num - 1]
                print(f"Usando experimento selecionado: {selected_experiment}")
            else:
                print(f"Número inválido. Usando experimento mais recente: {experiment_dirs[0]}")
                selected_experiment = experiment_dirs[0]
        except ValueError:
            print(f"Entrada inválida. Usando experimento mais recente: {experiment_dirs[0]}")
            selected_experiment = experiment_dirs[0]
    
    current_results_dir = os.path.join(results_dir, selected_experiment)
    experiment_dir = current_results_dir
    
    # Detectar modo (default ou optimized)
    if 'default' in selected_experiment:
        detected_mode = 'default'
    elif 'optimized' in selected_experiment:
        detected_mode = 'optimized'
    else:
        detected_mode = 'unknown'
    
    print(f"\nModo detectado: {detected_mode.upper()}")
    
    # Procurar modelos com nomes padronizados e variações
    model_variations = {
        'decision_tree': ['decision_tree'],
        'random_forest': ['random_forest'],
        'gradient_boosting': ['gradient_boosting'],
        'histogram_gradient_boosting': ['histogram_gradient_boosting'],
        'k_nearest_neighbors': ['k_nearest_neighbors', 'k-nearest_neighbors'],
        'multi_layer_perceptron': ['multi_layer_perceptron', 'multi-layer_perceptron'],
        'support_vector_classifier': ['support_vector_classifier', 'svc'],
        'catboost': ['catboost']
    }
    
    if detected_mode == 'default':
        # Modo default: procurar em pastas de modelos por default_metrics.json
        for standard_name, variations in model_variations.items():
            for model_name in variations:
                model_dir = os.path.join(current_results_dir, model_name)
                
                if os.path.exists(model_dir):
                    metrics_files = [
                        os.path.join(model_dir, 'default_metrics.json'),
                        os.path.join(model_dir, 'metrics.json'),
                    ]
                    
                    for metrics_file in metrics_files:
                        if os.path.exists(metrics_file):
                            print(f"Carregando {standard_name} de {os.path.basename(metrics_file)}...")
                            try:
                                with open(metrics_file, 'r') as f:
                                    data = json.load(f)
                                
                                if 'test_predictions' in data:
                                    predictions = data['test_predictions']
                                    if predictions and 'y_true' in predictions and 'y_pred_proba' in predictions:
                                        models_data[standard_name] = {
                                            'predictions': predictions,
                                            'file_path': metrics_file
                                        }
                                        print(f"  ✓ {standard_name}: {len(predictions['y_true'])} amostras")
                                        break
                                    else:
                                        print(f"  ⚠️  {standard_name}: test_predictions incompleto")
                                else:
                                    print(f"  ⚠️  {standard_name}: sem test_predictions")
                            except Exception as e:
                                print(f"  ❌ {standard_name}: {e}")
                            
                            if standard_name in models_data:
                                break
    
    elif detected_mode == 'optimized':
        # Modo optimized: procurar nested_cv_*.json em modelo folders (novo) ou no root (antigo)
        # Estratégia:
        # 1. Procurar em pastas de modelos por nested_cv_*.json (novo formato com modelo folder)
        # 2. Se não encontrado, procurar nested_cv_*.json no root do experimento (formato antigo)
        
        for standard_name, variations in model_variations.items():
            found = False
            
            # Tenta procurar em pasta de modelo (novo formato)
            for model_name in variations:
                model_dir = os.path.join(current_results_dir, model_name)
                if os.path.exists(model_dir):
                    # Procurar por metrics.json nesta pasta (novo formato otimizado)
                    metrics_file = os.path.join(model_dir, 'metrics.json')
                    if os.path.exists(metrics_file):
                        print(f"Carregando {standard_name} de metrics.json...")
                        try:
                            with open(metrics_file, 'r') as f:
                                data = json.load(f)
                            
                            # Tentar carregar test_predictions do novo formato
                            if 'test_predictions' in data and data['test_predictions']:
                                predictions = data['test_predictions']
                                if 'y_true' in predictions and 'y_pred_proba' in predictions:
                                    models_data[standard_name] = {
                                        'predictions': predictions,
                                        'file_path': metrics_file,
                                        'aggregated_metrics': data.get('aggregated_metrics', {})
                                    }
                                    print(f"  ✓ {standard_name}: {len(predictions['y_true'])} amostras (test_predictions)")
                                    found = True
                                    break
                            # Se não tem test_predictions, usar agregadas (fallback - sem curve real)
                            elif 'aggregated_metrics' in data:
                                agg = data['aggregated_metrics']
                                print(f"  ⚠️  {standard_name}: usando métricas agregadas (sem test_predictions)")
                                models_data[standard_name] = {
                                    'predictions': None,
                                    'file_path': metrics_file,
                                    'aggregated_metrics': agg
                                }
                                found = True
                                break
                        except Exception as e:
                            print(f"  ❌ Erro lendo metrics.json: {e}")
                
                if found:
                    break
            
            # Se não encontrou em pasta de modelo, procurar no root (formato antigo)
            if not found:
                try:
                    for file in os.listdir(current_results_dir):
                        if file.startswith('nested_cv_') and standard_name in file.lower() and file.endswith('.json'):
                            nested_cv_file = os.path.join(current_results_dir, file)
                            print(f"Carregando {standard_name} de {file} (root)...")
                            try:
                                with open(nested_cv_file, 'r') as f:
                                    data = json.load(f)
                                
                                if 'test_predictions' in data and data['test_predictions']:
                                    predictions = data['test_predictions']
                                    if 'y_true' in predictions and 'y_pred_proba' in predictions:
                                        models_data[standard_name] = {
                                            'predictions': predictions,
                                            'file_path': nested_cv_file,
                                            'aggregated_metrics': data.get('aggregated_metrics', {})
                                        }
                                        print(f"  ✓ {standard_name}: {len(predictions['y_true'])} amostras")
                                        found = True
                                        break
                                elif 'aggregated_metrics' in data:
                                    agg = data['aggregated_metrics']
                                    print(f"  ⚠️  {standard_name}: usando métricas agregadas (sem test_predictions)")
                                    models_data[standard_name] = {
                                        'predictions': None,
                                        'file_path': nested_cv_file,
                                        'aggregated_metrics': agg
                                    }
                                    found = True
                                    break
                            except Exception as e:
                                print(f"  ❌ Erro lendo {file}: {e}")
                            
                            if found:
                                break
                except Exception:
                    pass
    
    return models_data, experiment_dir, detected_mode


def detect_classification_type(y_true, y_pred_proba):
    """
    Detecta automaticamente se é classificação binária ou multiclasse
    
    Args:
        y_true: Array com labels verdadeiros
        y_pred_proba: Array com probabilidades preditas
    
    Returns:
        tuple: (classification_type, n_classes, class_names)
    """
    n_unique_classes = len(np.unique(y_true))
    
    # Verificar se y_pred_proba é 1D (binário) ou 2D (multiclasse)
    if len(np.array(y_pred_proba).shape) == 1:
        # Binário com apenas probabilidades da classe positiva
        return 'binary', 2, ['Passenger', 'Cancer Genes']
    else:
        y_pred_proba_array = np.array(y_pred_proba)
        n_prob_classes = y_pred_proba_array.shape[1] if len(y_pred_proba_array.shape) > 1 else 1
        
        if n_unique_classes == 2 and n_prob_classes == 2:
            return 'binary', 2, ['Passenger', 'Cancer Genes']
        elif n_unique_classes > 2 and n_prob_classes > 2:
            class_names = ['Passenger', 'TSG', 'Oncogenes'] if n_unique_classes == 3 else [f'Class {i}' for i in range(n_unique_classes)]
            return 'multiclass', n_unique_classes, class_names
        else:
            # Fallback para binário
            return 'binary', 2, ['Passenger', 'Cancer Genes']


def plot_roc_curve(model_results, save_path=None):
    """
    Cria gráfico de ROC Curve para todos os modelos usando predições salvas
    
    Args:
        model_results (dict): Resultados dos modelos com predições
        save_path (str): Caminho para salver o gráfico
    """
    plt.figure(figsize=(12, 9))
    
    models_plotted = 0
    models_with_errors = []
    classification_type = None
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name}...")
        
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])
            
            # Detectar tipo de classificação (apenas uma vez)
            if classification_type is None:
                classification_type, n_classes, class_names = detect_classification_type(y_true, y_pred_proba)
                print(f"  Tipo de classificação detectado: {classification_type} ({n_classes} classes)")
            
            model_display_name = format_model_name(model_name)
            color = STANDARD_COLORS.get(model_name.lower().replace('metrics', '').replace('_', '').replace('-', ''), f'C{models_plotted}')
            
            if classification_type == 'binary':
                # Classificação binária
                if len(y_pred_proba.shape) == 2:
                    # Se temos probabilidades para ambas as classes, usar a classe positiva
                    y_pred_proba_pos = y_pred_proba[:, 1]
                else:
                    # Se temos apenas probabilidades da classe positiva
                    y_pred_proba_pos = y_pred_proba
                
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba_pos)
                roc_auc = auc(fpr, tpr)
                
                # Check if we have stored metrics and compare
                if 'stored_metrics' in data and data['stored_metrics'].get('roc_auc') is not None:
                    stored_roc = data['stored_metrics']['roc_auc']
                    print(f"    ⚠️  Calculado: {roc_auc:.4f}, Arquivo: {stored_roc:.4f}, Diff: {abs(roc_auc - stored_roc):.4f}")
                    # Use the stored metric if available (it's the test set metric)
                    roc_auc = stored_roc
                
                plt.plot(fpr, tpr, color=color, lw=3, 
                        label=f'{model_display_name} (AUC = {roc_auc:.3f})')
                print(f"  {model_name}: AUC = {roc_auc:.3f}")

            models_plotted += 1
            
        except Exception as e:
            models_with_errors.append(model_name)
            print(f"  ❌ {model_name}: Erro ao processar predições - {e}")
    
    # Linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8, 
             label='Random Classifier (AUC = 0.500)')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    
    plt.legend(loc="lower right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + '_binary.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '_binary.pdf', bbox_inches='tight')
        print(f"ROC Curve salva em: " + save_path + '_binary')
    
    # Relatório final
    print(f"\nRelatório ROC:")
    print(f"  Tipo de classificação: {classification_type}")
    print(f"  Modelos plotados: {models_plotted}")
    if models_with_errors:
        print(f"  Modelos com erro: {', '.join(models_with_errors)}")


def plot_precision_recall_curve(model_results, save_path=None):
    """
    Cria gráfico de Precision-Recall Curve para todos os modelos usando predições salvas
    
    Args:
        model_results (dict): Resultados dos modelos com predições
        save_path (str): Caminho para salvar o gráfico
    """
    plt.figure(figsize=(12, 9))
    
    models_plotted = 0
    models_with_errors = []
    pos_rate = None
    classification_type = None
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name}...")
        
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])
            
            # Detectar tipo de classificação (apenas uma vez)
            if classification_type is None:
                classification_type, n_classes, class_names = detect_classification_type(y_true, y_pred_proba)
                print(f"  Tipo de classificação detectado: {classification_type} ({n_classes} classes)")
            
            model_display_name = format_model_name(model_name)
            color = STANDARD_COLORS.get(model_name.lower().replace('metrics', '').replace('_', '').replace('-', ''), f'C{models_plotted}')
            
            if classification_type == 'binary':
                # Classificação binária
                if pos_rate is None:
                    pos_rate = np.mean(y_true)
                
                if len(y_pred_proba.shape) == 2:
                    # Se temos probabilidades para ambas as classes, usar a classe positiva
                    y_pred_proba_pos = y_pred_proba[:, 1]
                else:
                    # Se temos apenas probabilidades da classe positiva
                    y_pred_proba_pos = y_pred_proba
                
                # Debug: verificar range de probabilidades
                print(f"    Probabilidades - min: {y_pred_proba_pos.min():.4f}, max: {y_pred_proba_pos.max():.4f}, mean: {y_pred_proba_pos.mean():.4f}")
                print(f"    Classes - positivos: {y_true.sum()}, negativos: {len(y_true) - y_true.sum()}")
                
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba_pos)
                # Use average_precision_score instead of auc for PR curves
                pr_auc = average_precision_score(y_true, y_pred_proba_pos)
                
                # Check if we have stored metrics and compare
                if 'stored_metrics' in data and data['stored_metrics'].get('pr_auc') is not None:
                    stored_pr = data['stored_metrics']['pr_auc']
                    print(f"    ⚠️  Calculado: {pr_auc:.4f}, Arquivo: {stored_pr:.4f}, Diff: {abs(pr_auc - stored_pr):.4f}")
                    # Use the stored metric if available (it's the test set metric)
                    pr_auc = stored_pr
                
                plt.plot(recall, precision, color=color, lw=3, 
                        label=f'{model_display_name} (AUC = {pr_auc:.3f})')
                print(f"  {model_name}: PR AUC = {pr_auc:.3f}")
            
            models_plotted += 1
            
        except Exception as e:
            models_with_errors.append(model_name)
            print(f"  ❌ {model_name}: Erro ao processar predições - {e}")
    
    # Linha base (classificador aleatório)
    if pos_rate is not None:
        plt.axhline(y=pos_rate, color='gray', lw=2, linestyle='--', alpha=0.8,
                    label=f'Random Classifier (AUC = {pos_rate:.3f})')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    
    plt.legend(loc="upper right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + '_binary.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '_binary.pdf', bbox_inches='tight')
        print(f"Precision-Recall Curve salva em: {save_path}")
        
    # Relatório final
    print(f"\nRelatório Precision-Recall:")
    print(f"  Modelos plotados: {models_plotted}")
    if models_with_errors:
        print(f"  Modelos com erro: {', '.join(models_with_errors)}")


def plot_confusion_matrices(model_results, save_path=None, class_names=None):
    """
    Cria gráficos de matriz de confusão para todos os modelos
    
    Args:
        model_results (dict): Resultados dos modelos com predições
        save_path (str): Caminho para salvar os gráficos
        class_names (list): Nomes das classes (opcional)
    """
    models_plotted = 0
    models_with_errors = []
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando matriz de confusão para {model_name}...")
        
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred = np.array(predictions.get('y_pred', []))
            
            # Se não tiver y_pred, usar y_pred_proba para gerar predições
            if len(y_pred) == 0:
                y_pred_proba = np.array(predictions['y_pred_proba'])
                if len(y_pred_proba.shape) == 2:
                    # Multiclasse ou binário com probabilidades de ambas as classes
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    # Binário com apenas probabilidade da classe positiva
                    y_pred = (y_pred_proba > 0.5).astype(int)
            
            model_display_name = format_model_name(model_name)
            
            # Calcular matriz de confusão
            cm = confusion_matrix(y_true, y_pred)
            
            # Criar figura
            plt.figure(figsize=(10, 8))
            
            # Usar seaborn para plotar heatmap
            if class_names is None:
                # Determinar se é binário ou multiclasse baseado no tamanho da matriz
                n_classes_cm = len(cm)
                if n_classes_cm == 2:
                    class_names = ['Passenger', 'Cancer Genes']
                elif n_classes_cm == 3:
                    class_names = ['Passenger', 'TSG', 'Oncogenes']
                else:
                    class_names = [f'Class {i}' for i in range(n_classes_cm)]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 22})
            
            plt.ylabel('True Label', fontsize=22)
            plt.xlabel('Predicted Label', fontsize=22)
            plt.tight_layout()
            
            if save_path:
                # Criar subpasta para confusion matrices
                cm_dir = os.path.dirname(save_path)
                cm_subdir = os.path.join(cm_dir, 'confusion_matrices')
                os.makedirs(cm_subdir, exist_ok=True)
                
                model_save_path = os.path.join(cm_subdir, f'cm_{model_name}')
                plt.savefig(model_save_path + '.png', dpi=300, bbox_inches='tight')
                plt.savefig(model_save_path + '.pdf', bbox_inches='tight')
                print(f"  Matriz de confusão salva: {model_save_path}")
            
            plt.close()
            models_plotted += 1
            
        except Exception as e:
            models_with_errors.append(model_name)
            print(f"  ❌ {model_name}: Erro ao processar - {e}")
    
    print(f"\nRelatório Confusion Matrices:")
    print(f"  Modelos plotados: {models_plotted}")
    if models_with_errors:
        print(f"  Modelos com erro: {', '.join(models_with_errors)}")


def browse_directories(base_dir="./results_final_tcc"):
    """
    Permite navegar pelos diretórios interativamente para selecionar de onde carregar os dados
    
    Returns:
        str: Caminho completo do diretório selecionado ou None se cancelado
    """
    current_dir = os.path.abspath(base_dir)
    
    while True:
        print(f"\n{'='*80}")
        print(f"Diretório atual: {current_dir}")
        print(f"{'='*80}")
        
        if not os.path.exists(current_dir):
            print(f"❌ Diretório não existe: {current_dir}")
            return None
        
        # Listar conteúdo do diretório
        try:
            items = os.listdir(current_dir)
            dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_dir, d))])
            files = sorted([f for f in items if f.endswith('.json')])
        except PermissionError:
            print("❌ Sem permissão para acessar este diretório")
            return None
        
        # Mostrar opções
        print("\nOpções:")
        print("0. Voltar ao diretório pai")
        print("S. Selecionar este diretório (USAR ESTE)")
        print("Q. Cancelar e voltar\n")
        
        # Mostrar subdiretórios
        if dirs:
            print("Subdiretórios:")
            for i, d in enumerate(dirs, 1):
                print(f"{i}. 📁 {d}")
        else:
            print("(Nenhum subdiretório)")
        
        # Mostrar arquivos JSON encontrados
        if files:
            print(f"\nArquivos JSON encontrados: {len(files)}")
            for f in files[:5]:  # Mostrar apenas os primeiros 5
                print(f"  • {f}")
            if len(files) > 5:
                print(f"  ... e mais {len(files) - 5} arquivos")
        
        # Solicitar escolha
        choice = input("\nEscolha uma opção: ").strip().upper()
        
        if choice == 'Q':
            print("Cancelado")
            return None
        elif choice == 'S':
            print(f"✅ Selecionado: {current_dir}")
            return current_dir
        elif choice == '0':
            # Voltar ao diretório pai
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Já está na raiz
                print("⚠️  Já está no diretório raiz")
            else:
                current_dir = parent_dir
        else:
            # Tentar converter para número e entrar no subdiretório
            try:
                dir_num = int(choice)
                if 1 <= dir_num <= len(dirs):
                    current_dir = os.path.join(current_dir, dirs[dir_num - 1])
                else:
                    print(f"❌ Número inválido. Escolha entre 1 e {len(dirs)}")
            except ValueError:
                print("❌ Opção inválida")


def generate_all_plots():
    """
    Gera todos os gráficos ROC e PR curves para os modelos testados.
    Carrega predições salvas dos arquivos metrics.json.
    """
    print("="*70, flush=True)
    print("Gerando gráficos de performance dos modelos usando predições salvas...", flush=True)
    print("="*70, flush=True)
    
    use_forplots = False
    forplots_path = None
    
    # Perguntar ao usuário qual modo usar
    print("\nEscolha o modo de carregamento:", flush=True)
    print("1. Experimentos normais (diretório results/)", flush=True)
    print("2. Navegar diretórios (escolher manualmente)", flush=True)
    print("", flush=True)
    
    mode_choice = input("Selecione uma opção [1-2] (Enter = 1): ").strip()
    
    if mode_choice == "2":
        # Modo navegação
        print("\n🗂️  Modo Navegação de Diretórios")
        selected_dir = browse_directories()
        
        if selected_dir:
            use_forplots = True
            forplots_path = selected_dir
            print(f"Usando: {forplots_path}")
        else:
            print("Voltando ao modo normal...")
            use_forplots = False
    else:
        print("Usando modo normal (experimentos)")
        use_forplots = False
    
    print("\nCarregando predições salvas dos modelos...")
    model_results, experiment_dir, detected_mode = load_saved_predictions(
        use_forplots=use_forplots, 
        forplots_path=forplots_path
    )
    
    if not model_results:
        print("❌ Nenhuma predição encontrada!")
        print("Execute primeiro os modelos usando main.py para gerar as predições.")
        return
    
    if not experiment_dir:
        print("❌ Nenhum diretório de experimento encontrado!")
        return
    
    # Filtrar modelos que têm test_predictions (caso haja alguns sem)
    models_with_predictions = {k: v for k, v in model_results.items() if v.get('predictions') is not None}
    
    if not models_with_predictions:
        print("❌ Nenhum modelo com test_predictions encontrado!")
        if detected_mode == 'optimized':
            print("⚠️  Modelos otimizados sem test_predictions não podem gerar curvas ROC/PR")
            print("    Você pode re-executar com a versão atualizada do código que salva test_predictions")
        return
    
    # Detectar tipo de classificação
    first_model = next(iter(models_with_predictions.values()))
    predictions = first_model['predictions']
    y_true = np.array(predictions['y_true'])
    y_pred_proba = np.array(predictions['y_pred_proba'])
    classification_type, n_classes, class_names = detect_classification_type(y_true, y_pred_proba)
    
    # Criar estrutura de diretórios
    curves_dir = os.path.join(experiment_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    
    models_loaded = len(models_with_predictions)
    models_skipped = len(model_results) - models_loaded
    print(f"✅ Predições encontradas para {models_loaded} modelos: {list(models_with_predictions.keys())}")
    if models_skipped > 0:
        print(f"⚠️  {models_skipped} modelos sem test_predictions foram ignorados")
    print(f"📊 Tipo de classificação: {classification_type} ({n_classes} classes)")
    print(f"🔄 Modo detectado: {detected_mode.upper()}")
    
    if classification_type == 'multiclass' and n_classes > 2:
        print(f"\n📈 Análise multiclasse detectada ({n_classes} classes)")
        generate_multiclass_plots(models_with_predictions, class_names, curves_dir)

    else:
        print(f"\n📈 Análise binária detectada")
        print("\nGerando ROC Curves...")
        roc_save_path = os.path.join(curves_dir, "roc_comparison")
        plot_roc_curve(models_with_predictions, save_path=roc_save_path)
        
        print("\nGerando PR Curves...")
        pr_save_path = os.path.join(curves_dir, "pr_comparison")
        plot_precision_recall_curve(models_with_predictions, save_path=pr_save_path)
    
    # Gerar matrizes de confusão para todos os tipos de classificação
    print("\nGerando Confusion Matrices...")
    cm_save_path = os.path.join(curves_dir, "confusion_matrices")
    plot_confusion_matrices(models_with_predictions, save_path=cm_save_path, class_names=class_names)
    
    print(f"\n🎉 Todos os gráficos gerados com sucesso!")

if __name__ == "__main__":
    print("\n" + "="*70, flush=True)
    print("INICIANDO GERAÇÃO DE GRÁFICOS", flush=True)
    print("="*70 + "\n", flush=True)
    generate_all_plots()
    print("\n" + "="*70, flush=True)
    print("PROCESSO FINALIZADO", flush=True)
    print("="*70 + "\n", flush=True)
