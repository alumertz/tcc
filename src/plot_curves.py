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
from datetime import datetime
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


def create_plots_directory():
    """Cria diretório para salvar os gráficos"""
    plots_dir = "../plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Criar subdiretórios para diferentes tipos de gráficos
    os.makedirs(os.path.join(plots_dir, "roc_curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "pr_curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "combined"), exist_ok=True)
    
    return plots_dir


def load_saved_predictions(results_dirs=None):
    """
    Carrega predições salvas de todos os modelos testados
    
    Args:
        results_dirs (list): Lista de diretórios para procurar resultados
        
    Returns:
        dict: Dicionário com predições de cada modelo
    """
    if results_dirs is None:
        # Procurar primeiro no novo formato (results/) depois no antigo (plot_results/)
        results_dirs = ["./results", "./plot_results"]
    
    models_data = {}
    
    # Lista de modelos disponíveis (incluindo variações de nomes com hífens e underscores)
    model_names = [
        'decision_tree', 'random_forest', 'gradient_boosting', 
        'histogram_gradient_boosting', 'k_nearest_neighbors', 'multi_layer_perceptron', 
        'support_vector_classifier', 'catboost',
        # Variações com hífens (formato usado pelos resultados salvos)
        'k-nearest_neighbors', 'multi-layer_perceptron'
    ]
    
    print("Procurando resultados em:")
    for results_dir in results_dirs:
        print(f"  - {results_dir}")
    print()
    
    # Procurar nas pastas de experimentos mais recentes primeiro
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            print(f"Diretório {results_dir} não existe")
            continue
            
        # Se for o novo formato results/, procurar por experimentos timestamped
        if results_dir == "./results":
            # Listar subdiretórios de experimentos (formato: YYYYMMDD_HHMMSS_ana_default_binary)
            experiment_dirs = []
            if os.path.exists(results_dir):
                for item in os.listdir(results_dir):
                    item_path = os.path.join(results_dir, item)
                    if os.path.isdir(item_path):
                        experiment_dirs.append(item)
            
            if experiment_dirs:
                # Usar o experimento mais recente
                latest_experiment = sorted(experiment_dirs)[-1]
                current_results_dir = os.path.join(results_dir, latest_experiment)
                print(f"Usando experimento mais recente: {latest_experiment}")
            else:
                print(f"Nenhum experimento encontrado em {results_dir}")
                continue
        else:
            current_results_dir = results_dir
        
        # Procurar modelos
        for model_name in model_names:
            if model_name in models_data:
                continue  # Já encontramos este modelo
                
            model_dir = os.path.join(current_results_dir, model_name)
            print(f"Verificando {model_name} em {model_dir}...")
            
            if os.path.exists(model_dir):
                # Procurar por arquivos de resultados (novos formatos sem timestamp e antigos com timestamp)
                json_files = [f for f in os.listdir(model_dir) if f.startswith('trials_') and f.endswith('.json')]
                default_files = [f for f in os.listdir(model_dir) 
                               if (f == 'default_metrics.json' or f.startswith('default_metrics_')) and f.endswith('.json')]
                
                latest_file = None
                file_type = None
                
                # Preferir arquivos sem timestamp (novos) sobre arquivos com timestamp (antigos)
                if json_files:
                    latest_file = sorted(json_files)[-1]
                    file_type = 'optimized'
                elif default_files:
                    # Priorizar default_metrics.json (novo formato) sobre default_metrics_TIMESTAMP.json (antigo)
                    if 'default_metrics.json' in default_files:
                        latest_file = 'default_metrics.json'
                    else:
                        latest_file = sorted(default_files)[-1]  # Pegar o mais recente com timestamp
                    file_type = 'default'
                
                if latest_file:
                    file_path = os.path.join(model_dir, latest_file)
                    
                    print(f"   Carregando predições de {model_name} ({file_type})")
                    print(f"   Arquivo: {latest_file}")
                    
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                        # Extrair predições baseado no tipo de arquivo
                        predictions = None
                        if file_type == 'optimized':
                            # Para arquivos otimizados, as predições estão em 'test_predictions'
                            if 'test_predictions' in data:
                                predictions = data['test_predictions']
                            # Fallback: pode estar diretamente no data se for formato antigo
                            elif isinstance(data, list) and len(data) > 0:
                                # Formato antigo de trials - não temos predições salvas
                                print(f"   ⚠️  {model_name}: Formato antigo sem predições salvas")
                                continue
                        elif file_type == 'default':
                            # Para arquivos default, as predições estão em 'test_predictions'
                            if 'test_predictions' in data:
                                predictions = data['test_predictions']
                        
                        if predictions and all(key in predictions for key in ['y_true', 'y_pred_proba']):
                            models_data[model_name] = {
                                'predictions': predictions,
                                'file_type': file_type,
                                'file_path': file_path
                            }
                            print(f"   ✓ Predições carregadas: {len(predictions['y_true'])} amostras")
                        else:
                            print(f"   ⚠️  {model_name}: Predições não encontradas no arquivo")
                            
                    except Exception as e:
                        print(f"   ❌ Erro ao carregar {model_name}: {e}")
                else:
                    print(f"   ⚠️  Nenhum arquivo de resultados encontrado para {model_name}")
            else:
                print(f"   ⚠️  Diretório não existe: {model_dir}")
    
    return models_data


def plot_roc_curve(model_results, save_path=None):
    """
    Cria gráfico de ROC Curve para todos os modelos usando predições salvas
    
    Args:
        model_results (dict): Resultados dos modelos com predições
        save_path (str): Caminho para salver o gráfico
    """
    plt.figure(figsize=(12, 9))
    
    # Cores específicas para cada modelo para melhor visualização
    model_colors = {
        'decision_tree': '#1f77b4',      # Azul
        'random_forest': '#ff7f0e',      # Laranja
        'gradient_boosting': '#2ca02c',  # Verde
        'histogram_gradient_boosting': '#d62728',  # Vermelho
        'k_nearest_neighbors': '#9467bd',                # Roxo
        'multi_layer_perceptron': '#8c564b',                # Marrom
        'support_vector_classifier': '#e377c2',                 # Rosa
        'catboost': '#17becf',           # Ciano
    }
    
    models_plotted = 0
    models_with_errors = []
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name}...")
        
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])
            
            # Calcular ROC
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plotar curva
            model_display_name = model_name.replace('_', ' ').title()
            color = model_colors.get(model_name, f'C{models_plotted}')
            plt.plot(fpr, tpr, color=color, lw=3, 
                    label=f'{model_display_name} (AUC = {roc_auc:.3f})')
            models_plotted += 1
            print(f"  {model_name}: AUC = {roc_auc:.3f}")
            
        except Exception as e:
            models_with_errors.append(model_name)
            print(f"  ❌ {model_name}: Erro ao processar predições - {e}")
    
    # Linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8, 
             label='(AUC = 0.500)')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    plt.legend(loc="lower right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path+'.pdf', bbox_inches='tight')
        print(f"ROC Curve salva em: {save_path}")
    
    plt.show()
    
    # Relatório final
    print(f"\nRelatório ROC:")
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
    
    # Cores específicas para cada modelo para melhor visualização
    model_colors = {
        'decision_tree': '#1f77b4',      # Azul
        'random_forest': '#ff7f0e',      # Laranja
        'gradient_boosting': '#2ca02c',  # Verde
        'histogram_gradient_boosting': '#d62728',  # Vermelho
        'k_nearest_neighbors': '#9467bd',                # Roxo
        'multi_layer_perceptron': '#8c564b',                # Marrom
        'support_vector_classifier': '#e377c2',                 # Rosa
        'catboost': '#17becf',           # Ciano
    }
    
    models_plotted = 0
    models_with_errors = []
    pos_rate = None
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name}...")
        
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])
            
            # Calcular proporção da classe positiva (apenas uma vez)
            if pos_rate is None:
                pos_rate = np.mean(y_true)
            
            # Calcular Precision-Recall
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Plotar curva
            model_display_name = model_name.replace('_', ' ').title()
            color = model_colors.get(model_name, f'C{models_plotted}')
            plt.plot(recall, precision, color=color, lw=3, 
                    label=f'{model_display_name} (AUC = {pr_auc:.3f})')
            models_plotted += 1
            print(f"  {model_name}: PR AUC = {pr_auc:.3f}")
            
        except Exception as e:
            models_with_errors.append(model_name)
            print(f"  ❌ {model_name}: Erro ao processar predições - {e}")
    
    # Linha base (classificador aleatório)
    if pos_rate is not None:
        plt.axhline(y=pos_rate, color='gray', lw=2, linestyle='--', alpha=0.8,
                    label=f'(AUC = {pos_rate:.3f})')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall ', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.legend(loc="upper right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path+'.pdf', bbox_inches='tight')
        print(f"Precision-Recall Curve salva em: {save_path}")
    
    plt.show()
    
    # Relatório final
    print(f"\nRelatório Precision-Recall:")
    print(f"  Modelos plotados: {models_plotted}")
    if models_with_errors:
        print(f"  Modelos com erro: {', '.join(models_with_errors)}")


def plot_combined_curves(model_results, save_path=None):
    """
    Cria gráfico combinado com ROC e Precision-Recall lado a lado usando predições salvas
    
    Args:
        model_results (dict): Resultados dos modelos com predições
        save_path (str): Caminho para salvar o gráfico
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Cores específicas para cada modelo
    model_colors = {
        'decision_tree': '#1f77b4',      # Azul
        'random_forest': '#ff7f0e',      # Laranja
        'gradient_boosting': '#2ca02c',  # Verde
        'histogram_gradient_boosting': '#d62728',  # Vermelho
        'k_nearest_neighbors': '#9467bd',                # Roxo
        'multi_layer_perceptron': '#8c564b',                # Marrom
        'support_vector_classifier': '#e377c2',                 # Rosa
        'catboost': '#17becf',           # Ciano
    }
    
    models_plotted = 0
    models_with_errors = []
    pos_rate = None
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name} (gráfico combinado)...")
        
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])
            
            # Calcular proporção da classe positiva (apenas uma vez)
            if pos_rate is None:
                pos_rate = np.mean(y_true)
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Model display name
            model_display_name = model_name.replace('_', ' ').title()
            color = model_colors.get(model_name, f'C{models_plotted}')
            
            # Plot ROC
            ax1.plot(fpr, tpr, color=color, lw=3, 
                    label=f'{model_display_name} (AUC = {roc_auc:.3f})')
            
            # Plot PR
            ax2.plot(recall, precision, color=color, lw=3, 
                    label=f'{model_display_name} (AUC = {pr_auc:.3f})')
            
            models_plotted += 1
            print(f"  {model_name}: ROC AUC = {roc_auc:.3f}, PR AUC = {pr_auc:.3f}")
            
        except Exception as e:
            models_with_errors.append(model_name)
            print(f"  ❌ {model_name}: Erro ao processar predições - {e}")
    
    # ROC subplot
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8,
             label='Classificador Aleatório (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
    ax1.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    ax1.set_title('Curvas ROC', fontsize=14)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # PR subplot
    if pos_rate is not None:
        ax2.axhline(y=pos_rate, color='gray', lw=2, linestyle='--', alpha=0.8,
                    label=f'Classificador Aleatório (AUC = {pos_rate:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=16)
    ax2.set_ylabel('Precision', fontsize=16)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path+'.pdf', bbox_inches='tight')
        print(f"Gráfico combinado salvo em: {save_path}")
    
    plt.show()
    
    # Relatório final
    print(f"\nRelatório Gráfico Combinado:")
    print(f"  Modelos plotados: {models_plotted}")
    if models_with_errors:
        print(f"  Modelos com erro: {', '.join(models_with_errors)}")


def generate_all_plots():
    """
    Função principal para gerar todos os gráficos usando predições salvas
    """
    print("Gerando gráficos de performance dos modelos usando predições salvas...")
    print("="*70)
    
    # Criar diretório de plots
    plots_dir = create_plots_directory()
    
    # Carregar predições salvas
    print("Carregando predições salvas dos modelos...")
    model_results = load_saved_predictions()
    
    if not model_results:
        print("❌ Nenhuma predição encontrada!")
        print("Execute primeiro os modelos usando main.py para gerar as predições.")
        return
    
    print(f"✅ Predições encontradas para {len(model_results)} modelos: {list(model_results.keys())}")
    
    # Timestamp para arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Gerar gráficos
    print("\nGerando ROC Curves...")
    roc_save_path = os.path.join(plots_dir, "roc_curves", f"roc_comparison_{timestamp}")
    plot_roc_curve(model_results, save_path=roc_save_path)
    
    print("\nGerando Precision-Recall Curves...")
    pr_save_path = os.path.join(plots_dir, "pr_curves", f"pr_comparison_{timestamp}")
    plot_precision_recall_curve(model_results, save_path=pr_save_path)
    
    
    print(f"\n🎉 Todos os gráficos gerados com sucesso!")
    print(f"📁 Gráficos salvos em: {plots_dir}")

if __name__ == "__main__":
    # Exemplo de uso
    print("Executando geração de gráficos...")
    generate_all_plots()
