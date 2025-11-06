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
from sklearn.metrics import roc_curve, precision_recall_curve, auc
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
    'decision_tree': '#1f77b4',           # Azul
    'random_forest': '#ff7f0e',           # Laranja
    'gradient_boosting': '#2ca02c',       # Verde
    'histogram_gradient_boosting': '#d62728',  # Vermelho
    'k_nearest_neighbors': '#9467bd',     # Roxo
    'multi_layer_perceptron': '#8c564b',  # Marrom
    'support_vector_classifier': '#e377c2',  # Rosa
    'catboost': '#17becf',                # Ciano
}


def create_plots_directory(experiment_dir):
    """Cria diretório curves dentro do diretório do experimento para salvar os gráficos"""
    curves_dir = os.path.join(experiment_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    
    return curves_dir


def load_saved_predictions(results_dir="./results"):
    """
    Carrega predições salvas de todos os modelos testados
    
    Args:
        results_dir (str): Diretório para procurar resultados
        
    Returns:
        tuple: (models_data, experiment_dir) - Dicionário com predições e caminho do experimento
    """
    models_data = {}
    experiment_dir = None
    
    # Lista de modelos padronizada (sem variações)
    model_names = [
        'decision_tree', 'random_forest', 'gradient_boosting', 
        'histogram_gradient_boosting', 'k_nearest_neighbors', 'multi_layer_perceptron', 
        'support_vector_classifier', 'catboost'
    ]
    
    print(f"Procurando resultados em: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"Diretório {results_dir} não existe")
        return models_data, experiment_dir
    
    # Listar subdiretórios de experimentos (formato: YYYYMMDD_HHMMSS_ana_default_*)
    experiment_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Verificar se segue o padrão de timestamp de experimento
            if len(item) > 15 and item[8] == '_' and item[15] == '_' and 'ana_default' in item:
                experiment_dirs.append(item)
    
    if not experiment_dirs:
        print(f"Nenhum experimento encontrado em {results_dir}")
        print(f"Diretórios disponíveis: {os.listdir(results_dir)}")
        return models_data, experiment_dir
    
    # Ordenar diretórios por timestamp (mais recente primeiro)
    experiment_dirs.sort(reverse=True)
    
    # Mostrar opções para o usuário
    print("\nDiretórios de experimentos disponíveis:")
    for i, dir_name in enumerate(experiment_dirs, 1):
        # Extrair informações do nome do diretório para melhor visualização
        parts = dir_name.split('_')
        if len(parts) >= 4:
            date_part = parts[0]  # YYYYMMDD
            time_part = parts[1]  # HHMMSS
            classification_type = parts[-1]  # binary/multiclass
            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
            print(f"{i}. {dir_name}")
            print(f"   Data: {formatted_date} {formatted_time} | Tipo: {classification_type}")
        else:
            print(f"{i}. {dir_name}")
    
    # Solicitar seleção do usuário
    print(f"\n[Enter] = usar o mais recente ({experiment_dirs[0]})")
    choice = input("Selecione o número do experimento (ou pressione Enter): ").strip()
    
    if choice == "":
        # Usar o experimento mais recente (primeiro da lista ordenada)
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
    experiment_dir = current_results_dir  # Armazenar o caminho do experimento
    
    # Procurar modelos com nomes padronizados e variações (para compatibilidade)
    model_variations = {
        'decision_tree': ['decision_tree'],
        'random_forest': ['random_forest'],
        'gradient_boosting': ['gradient_boosting'],
        'histogram_gradient_boosting': ['histogram_gradient_boosting'],
        'k_nearest_neighbors': ['k_nearest_neighbors', 'k-nearest_neighbors'],
        'multi_layer_perceptron': ['multi_layer_perceptron', 'multi-layer_perceptron'],
        'support_vector_classifier': ['support_vector_classifier'],
        'catboost': ['catboost']
    }
    
    for standard_name, variations in model_variations.items():
        for model_name in variations:
            model_dir = os.path.join(current_results_dir, model_name)
            print(f"Verificando {model_name} em {model_dir}...")
            
            if os.path.exists(model_dir):
                # Procurar por arquivos metrics.json (com diferentes prefixos)
                metrics_files = [
                    os.path.join(model_dir, 'metrics.json'),          # Padrão sem prefixo
                    os.path.join(model_dir, 'default_metrics.json'),  # Com prefixo default_
                ]
                
                metrics_file = None
                for mf in metrics_files:
                    if os.path.exists(mf):
                        metrics_file = mf
                        break
                
                if metrics_file:
                    print(f"   Carregando predições de {standard_name}")
                    print(f"   Arquivo: {os.path.basename(metrics_file)}")
                    
                    try:
                        with open(metrics_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extrair predições
                        if 'test_predictions' in data:
                            predictions = data['test_predictions']
                            
                            if predictions and all(key in predictions for key in ['y_true', 'y_pred_proba']):
                                models_data[standard_name] = {
                                    'predictions': predictions,
                                    'file_path': metrics_file
                                }
                                print(f"   ✓ Predições carregadas: {len(predictions['y_true'])} amostras")
                                break  # Encontrou o modelo, não precisa procurar outras variações
                            else:
                                print(f"   ⚠️  {standard_name}: Predições não encontradas no arquivo")
                        else:
                            print(f"   ⚠️  {standard_name}: Campo 'test_predictions' não encontrado")
                            
                    except Exception as e:
                        print(f"   ❌ Erro ao carregar {standard_name}: {e}")
                else:
                    print(f"   ⚠️  Nenhum arquivo de métricas encontrado para {model_name}")
                    print(f"       Procurou por: metrics.json, default_metrics.json")
            else:
                print(f"   ⚠️  Diretório não existe: {model_dir}")
    
    return models_data, experiment_dir


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
        return 'binary', 2, ['Class 0', 'Class 1']
    else:
        y_pred_proba_array = np.array(y_pred_proba)
        n_prob_classes = y_pred_proba_array.shape[1] if len(y_pred_proba_array.shape) > 1 else 1
        
        if n_unique_classes == 2 and n_prob_classes == 2:
            return 'binary', 2, ['Class 0', 'Class 1']
        elif n_unique_classes > 2 and n_prob_classes > 2:
            class_names = [f'Class {i}' for i in range(n_unique_classes)]
            return 'multiclass', n_unique_classes, class_names
        else:
            # Fallback para binário
            return 'binary', 2, ['Class 0', 'Class 1']


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
            
            model_display_name = model_name.replace('_', ' ').title()
            color = STANDARD_COLORS.get(model_name, f'C{models_plotted}')
            
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
    
    plt.title('ROC Curves (Binary Classification)', fontsize=24)
    
    plt.legend(loc="lower right", fontsize=18, framealpha=0.9)
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
            
            model_display_name = model_name.replace('_', ' ').title()
            color = STANDARD_COLORS.get(model_name, f'C{models_plotted}')
            
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
                
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba_pos)
                pr_auc = auc(recall, precision)
                
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
    
    plt.title('Precision-Recall Curves (Binary Classification)', fontsize=24)
    
    plt.legend(loc="lower left", fontsize=18, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + '_binary.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '_binary.pdf', bbox_inches='tight')
        print(f"Precision-Recall Curve salva em: {save_path}{suffix}")
        
    # Relatório final
    print(f"\nRelatório Precision-Recall:")
    print(f"  Modelos plotados: {models_plotted}")
    if models_with_errors:
        print(f"  Modelos com erro: {', '.join(models_with_errors)}")


def generate_all_plots():
    """
    Função principal para gerar todos os gráficos usando predições salvas
    """
    print("Gerando gráficos de performance dos modelos usando predições salvas...")
    print("="*70)
    
    print("Carregando predições salvas dos modelos...")
    model_results, experiment_dir = load_saved_predictions()
    
    if not model_results:
        print("❌ Nenhuma predição encontrada!")
        print("Execute primeiro os modelos usando main.py para gerar as predições.")
        return
    
    if not experiment_dir:
        print("❌ Nenhum diretório de experimento encontrado!")
        return
    
    # Detectar tipo de classificação
    first_model = next(iter(model_results.values()))
    predictions = first_model['predictions']
    y_true = np.array(predictions['y_true'])
    y_pred_proba = np.array(predictions['y_pred_proba'])
    classification_type, n_classes, class_names = detect_classification_type(y_true, y_pred_proba)
    
    # Criar estrutura de diretórios
    curves_dir = os.path.join(experiment_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    
    print(f"✅ Predições encontradas para {len(model_results)} modelos: {list(model_results.keys())}")
    print(f"📊 Tipo de classificação: {classification_type} ({n_classes} classes)")
    
    if classification_type == 'multiclass' and n_classes > 2:
        print(f"\n📈 Análise multiclasse detectada ({n_classes} classes)")
        generate_multiclass_plots(model_results, class_names, curves_dir)

    else:
        print(f"\n📈 Análise binária detectada")
        print("\nGerando ROC Curves...")
        roc_save_path = os.path.join(curves_dir, "roc_comparison")
        plot_roc_curve(model_results, save_path=roc_save_path)
        
        print("\nGerando PR Curves...")
        pr_save_path = os.path.join(curves_dir, "pr_comparison")
        plot_precision_recall_curve(model_results, save_path=pr_save_path)
    
    print(f"\n🎉 Todos os gráficos gerados com sucesso!")

if __name__ == "__main__":
    print("Executando geração de gráficos...")
    generate_all_plots()
