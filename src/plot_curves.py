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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    plots_dir = "artigo/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Criar subdiretórios para diferentes tipos de gráficos
    os.makedirs(os.path.join(plots_dir, "roc_curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "pr_curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "combined"), exist_ok=True)
    
    return plots_dir


def load_model_results(results_dir="artigo/results"):
    """
    Carrega os resultados de todos os modelos testados
    
    Returns:
        dict: Dicionário com resultados de cada modelo
    """
    models_data = {}
    
    # Lista de modelos disponíveis
    model_names = [
        'decision_tree', 'random_forest', 'gradient_boosting', 
        'hist_gradient_boosting', 'knn', 'mlp', #'svc'
    ]
    
    for model_name in model_names:
        model_dir = os.path.join(results_dir, model_name)
        if os.path.exists(model_dir):
            # Encontrar o arquivo de teste mais recente
            test_files = [f for f in os.listdir(model_dir) if f.startswith('test_results_')]
            if test_files:
                latest_file = sorted(test_files)[-1]
                test_file_path = os.path.join(model_dir, latest_file)
                
                # Carregar dados do modelo
                models_data[model_name] = {
                    'test_file': test_file_path,
                    'model_dir': model_dir
                }
                
                # Tentar carregar arquivo JSON com trials
                json_files = [f for f in os.listdir(model_dir) if f.startswith('trials_')]
                if json_files:
                    latest_json = sorted(json_files)[-1]
                    json_file_path = os.path.join(model_dir, latest_json)
                    try:
                        with open(json_file_path, 'r') as f:
                            models_data[model_name]['trials'] = json.load(f)
                    except:
                        pass
    
    return models_data


def get_model_predictions(model_name, best_params, X, y):
    """
    Treina um modelo com os melhores parâmetros e retorna predições para gráficos
    
    Args:
        model_name (str): Nome do modelo
        best_params (dict): Melhores hiperparâmetros
        X (array): Features
        y (array): Labels
        
    Returns:
        tuple: (y_test, y_pred_proba)
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    
    # Mapeamento de modelos
    model_classes = {
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'hist_gradient_boosting': HistGradientBoostingClassifier,
        'knn': KNeighborsClassifier,
        'mlp': MLPClassifier,
        'svc': SVC
    }
    
    if model_name not in model_classes:
        return None, None
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Configurar parâmetros específicos
    params = best_params.copy()
    if model_name == 'svc':
        params['probability'] = True
        params['random_state'] = 30
    elif model_name in ['decision_tree', 'random_forest', 'gradient_boosting', 'mlp']:
        params['random_state'] = 30
    
    # Criar e treinar modelo
    model_class = model_classes[model_name]
    
    try:
        # Para MLP, reconstruir hidden_layer_sizes se necessário
        if model_name == 'mlp' and 'n_layers' in params:
            n_layers = params.pop('n_layers')
            hidden_layer_sizes = []
            for i in range(n_layers):
                if f'layer_{i}_size' in params:
                    hidden_layer_sizes.append(params.pop(f'layer_{i}_size'))
            params['hidden_layer_sizes'] = tuple(hidden_layer_sizes)
        
        # Criar pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model_class(**params))
        ])
        
        # Treinar
        pipeline.fit(X_train, y_train)
        
        # Predições
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        return y_test, y_pred_proba
        
    except Exception as e:
        print(f"Erro ao treinar {model_name}: {e}")
        return None, None


def plot_roc_curve(model_results, X, y, save_path=None):
    """
    Cria gráfico de ROC Curve para todos os modelos
    
    Args:
        model_results (dict): Resultados dos modelos
        X (array): Features
        y (array): Labels
        save_path (str): Caminho para salvar o gráfico
    """
    plt.figure(figsize=(12, 9))
    
    # Cores específicas para cada modelo para melhor visualização
    model_colors = {
        'decision_tree': '#1f77b4',      # Azul
        'random_forest': '#ff7f0e',      # Laranja
        'gradient_boosting': '#2ca02c',  # Verde
        'hist_gradient_boosting': '#d62728',  # Vermelho
        'knn': '#9467bd',                # Roxo
        'mlp': '#8c564b',                # Marrom
        'svc': '#e377c2'                 # Rosa
    }
    
    models_plotted = 0
    models_with_errors = []
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name}...")
        
        # Extrair melhores parâmetros dos trials
        if 'trials' in data and data['trials']:
            # Encontrar o melhor trial
            best_trial = max(data['trials'], key=lambda x: x.get('score', 0))
            best_params = best_trial.get('params', {})
            
            # Obter predições
            y_test, y_pred_proba = get_model_predictions(model_name, best_params, X, y)
            
            if y_test is not None and y_pred_proba is not None:
                # Calcular ROC
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plotar curva
                model_display_name = model_name.replace('_', ' ').title()
                color = model_colors.get(model_name, f'C{models_plotted}')
                plt.plot(fpr, tpr, color=color, lw=3, 
                        label=f'{model_display_name} (AUC = {roc_auc:.3f})')
                models_plotted += 1
                print(f"  {model_name}: AUC = {roc_auc:.3f}")
            else:
                models_with_errors.append(model_name)
                print(f"  {model_name}: Erro ao obter predições")
        else:
            models_with_errors.append(model_name)
            print(f"  {model_name}: Dados de trials não encontrados")
    
    # Linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8, 
             label='(AUC = 0.500)')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    #plt.title('Curvas ROC - Comparação entre Modelos\nClassificação de Genes-Alvo (Oncogenes)', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Adicionar informações
    # info_text = f'Dataset: {X.shape[0]} amostras, {X.shape[1]} features\nModelos plotados: {models_plotted}'
    # if models_with_errors:
    #     info_text += f'\nModelos com erro: {len(models_with_errors)}'
    
    # plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
    #          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    #          verticalalignment='top', fontsize=9)
    
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


def plot_precision_recall_curve(model_results, X, y, save_path=None):
    """
    Cria gráfico de Precision-Recall Curve para todos os modelos
    
    Args:
        model_results (dict): Resultados dos modelos
        X (array): Features
        y (array): Labels
        save_path (str): Caminho para salvar o gráfico
    """
    plt.figure(figsize=(12, 9))
    
    # Cores específicas para cada modelo para melhor visualização
    model_colors = {
        'decision_tree': '#1f77b4',      # Azul
        'random_forest': '#ff7f0e',      # Laranja
        'gradient_boosting': '#2ca02c',  # Verde
        'hist_gradient_boosting': '#d62728',  # Vermelho
        'knn': '#9467bd',                # Roxo
        'mlp': '#8c564b',                # Marrom
        'svc': '#e377c2'                 # Rosa
    }
    
    # Linha base (proporção da classe positiva)
    pos_rate = np.mean(y)
    
    models_plotted = 0
    models_with_errors = []
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name}...")
        
        # Extrair melhores parâmetros dos trials
        if 'trials' in data and data['trials']:
            # Encontrar o melhor trial
            best_trial = max(data['trials'], key=lambda x: x.get('score', 0))
            best_params = best_trial.get('params', {})
            
            # Obter predições
            y_test, y_pred_proba = get_model_predictions(model_name, best_params, X, y)
            
            if y_test is not None and y_pred_proba is not None:
                # Calcular Precision-Recall
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                # Plotar curva
                model_display_name = model_name.replace('_', ' ').title()
                color = model_colors.get(model_name, f'C{models_plotted}')
                plt.plot(recall, precision, color=color, lw=3, 
                        label=f'{model_display_name} (AUC = {pr_auc:.3f})')
                models_plotted += 1
                print(f"  {model_name}: PR AUC = {pr_auc:.3f}")
            else:
                models_with_errors.append(model_name)
                print(f"  {model_name}: Erro ao obter predições")
        else:
            models_with_errors.append(model_name)
            print(f"  {model_name}: Dados de trials não encontrados")
    
    # Linha base (classificador aleatório)
    plt.axhline(y=pos_rate, color='gray', lw=2, linestyle='--', alpha=0.8,
                label=f'(AUC = {pos_rate:.3f})')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall ', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    #plt.title('Curvas Precision-Recall - Comparação entre Modelos\nClassificação de Genes-Alvo (Oncogenes)', fontsize=14, pad=20)
    plt.legend(loc="upper right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Adicionar informações
    # info_text = f'Dataset: {X.shape[0]} amostras, {X.shape[1]} features\nClasse positiva: {pos_rate:.1%}\nModelos plotados: {models_plotted}'
    # if models_with_errors:
    #     info_text += f'\nModelos com erro: {len(models_with_errors)}'
    
    # plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
    #          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
    #          verticalalignment='top', fontsize=9)
    
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


def plot_combined_curves(model_results, X, y, save_path=None):
    """
    Cria gráfico combinado com ROC e Precision-Recall lado a lado
    
    Args:
        model_results (dict): Resultados dos modelos
        X (array): Features
        y (array): Labels
        save_path (str): Caminho para salvar o gráfico
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Cores específicas para cada modelo
    model_colors = {
        'decision_tree': '#1f77b4',      # Azul
        'random_forest': '#ff7f0e',      # Laranja
        'gradient_boosting': '#2ca02c',  # Verde
        'hist_gradient_boosting': '#d62728',  # Vermelho
        'knn': '#9467bd',                # Roxo
        'mlp': '#8c564b',                # Marrom
        'svc': '#e377c2'                 # Rosa
    }
    
    # Linha base para PR
    pos_rate = np.mean(y)
    
    models_plotted = 0
    models_with_errors = []
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name} (gráfico combinado)...")
        
        # Extrair melhores parâmetros dos trials
        if 'trials' in data and data['trials']:
            # Encontrar o melhor trial
            best_trial = max(data['trials'], key=lambda x: x.get('score', 0))
            best_params = best_trial.get('params', {})
            
            # Obter predições
            y_test, y_pred_proba = get_model_predictions(model_name, best_params, X, y)
            
            if y_test is not None and y_pred_proba is not None:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
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
            else:
                models_with_errors.append(model_name)
                print(f"  {model_name}: Erro ao obter predições")
        else:
            models_with_errors.append(model_name)
            print(f"  {model_name}: Dados de trials não encontrados")
    
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
    ax2.axhline(y=pos_rate, color='gray', lw=2, linestyle='--', alpha=0.8,
                label=f'Classificador Aleatório (AUC = {pos_rate:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=16)
    ax2.set_ylabel('Precision', fontsize=16)
    #ax2.set_title('Curvas Precision-Recal', fontsize=14)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Título geral
    fig.suptitle('Comparação de Performance - Classificação de Genes-Alvo (Oncogenes)', 
                 fontsize=16, y=1.02)
    
    # Adicionar informações
    info_text = f'Dataset: {X.shape[0]} amostras, {X.shape[1]} features\nModelos plotados: {models_plotted}'
    if models_with_errors:
        info_text += f'\nModelos com erro: {len(models_with_errors)}'
    
    fig.text(0.02, 0.02, info_text, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
             fontsize=9)
    
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


def generate_all_plots(X=None, y=None):
    """
    Função principal para gerar todos os gráficos
    
    Args:
        X (array, optional): Features. Se None, carrega dados do processamento
        y (array, optional): Labels. Se None, carrega dados do processamento
    """
    print("Gerando gráficos de performance dos modelos...")
    print("="*60)
    
    # Criar diretório de plots
    plots_dir = create_plots_directory()
    
    # Carregar dados se não fornecidos
    if X is None or y is None:
        try:
            # Usar os mesmos caminhos e função de carregamento do main.py
            features_path = "renan/data_files/omics_features/UNION_features.tsv"
            labels_path = "renan/data_files/labels/UNION_labels.tsv"
            
            # Verificar se os arquivos existem
            if not os.path.exists(features_path):
                print(f"Arquivo de features não encontrado: {features_path}")
                return
            
            if not os.path.exists(labels_path):
                print(f"Arquivo de labels não encontrado: {labels_path}")
                return
            
            from processing import prepare_dataset
            print("Carregando dados usando a função padrão do processamento...")
            
            X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path)
            
            if X is None:
                print("Erro ao preparar dataset usando processamento.py")
                return
                
            print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
            print(f"Distribuição das classes: {dict(zip(*np.unique(y, return_counts=True)))}")
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            print("Verifique se os arquivos de dados estão disponíveis")
            print("Ou forneça X e y como parâmetros para a função")
            return
    
    # Carregar resultados dos modelos
    print("Carregando resultados dos modelos...")
    model_results = load_model_results()
    
    if not model_results:
        print("Nenhum resultado de modelo encontrado!")
        print("Execute primeiro os modelos usando main.py")
        return
    
    print(f"Modelos encontrados: {list(model_results.keys())}")
    
    # Timestamp para arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Gerar gráficos
    print("\nGerando ROC Curves...")
    roc_save_path = os.path.join(plots_dir, "roc_curves", f"roc_comparison_{timestamp}")
    plot_roc_curve(model_results, X, y, save_path=roc_save_path)
    
    print("\nGerando Precision-Recall Curves...")
    pr_save_path = os.path.join(plots_dir, "pr_curves", f"pr_comparison_{timestamp}")
    plot_precision_recall_curve(model_results, X, y, save_path=pr_save_path)
    
    # print("\nGerando gráfico combinado...")
    # combined_save_path = os.path.join(plots_dir, "combined", f"combined_curves_{timestamp}")
    # plot_combined_curves(model_results, X, y, save_path=combined_save_path)
    
    print(f"\nTodos os gráficos gerados com sucesso!")
    print(f"Gráficos salvos em: {plots_dir}")
    print(f"Timestamp: {timestamp}")


if __name__ == "__main__":
    # Exemplo de uso
    print("Executando geração de gráficos...")
    generate_all_plots()


    
