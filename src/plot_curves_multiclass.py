#!/usr/bin/env python3
"""
Funções específicas para criar gráficos de ROC Curve e Precision-Recall Curve
para classificação multiclasse usando estratégias One-vs-Rest (OvR) e One-vs-One (OvO).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle, combinations
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
plt.rcParams['legend.fontsize'] = 22

# Configuração do seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

# Paleta de cores padronizada para todos os modelos (rainbow colors)
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

# Cores adicionais para classes em gráficos multiclasse (rainbow)
MULTICLASS_COLORS = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3', '#FF1493']

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

##### AUC-PR ######

def plot_pr_individual_models(model_results, curves_dir, n_classes, class_names):
    """
    Cria gráficos PR individuais para cada modelo
    Salva em pastas organizadas por nome do modelo dentro de curves/
    """
    for model_name, data in model_results.items():
        try:
            # Criar pasta para o modelo dentro de curves/
            model_curves_dir = os.path.join(curves_dir, model_name)
            os.makedirs(model_curves_dir, exist_ok=True)
            
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])
            
            # Binarize labels for multiclass PR curves
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            
            # Handle binary case
            if n_classes == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            
            plt.figure(figsize=(10, 8))
            colors = cycle(MULTICLASS_COLORS)
            
            for i in range(n_classes):
                color = next(colors)
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
                plt.plot(recall, precision, color=color, lw=2, 
                        label=f"{class_names[i]} (AP={ap:.3f})")
            
            # Add weighted average
            weighted_ap = average_precision_score(y_true_bin, y_pred_proba, average='weighted')
            
            model_display_name = format_model_name(model_name)
            plt.xlabel("Recall", fontsize=22)
            plt.ylabel("Precision", fontsize=22)
            plt.legend(loc="lower left", fontsize=22)
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.tight_layout()
            
            save_path = os.path.join(model_curves_dir, f"{model_name}_pr_curves")
            plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
            plt.savefig(save_path + '.pdf', bbox_inches='tight')
            plt.close()
            
            print(f"PR curves salvo para {model_name} em {model_curves_dir}")
        except Exception as e:
            print(f"Erro gerando PR curves para {model_name}: {e}")
            plt.close()


######## Weighted AUC PR and ROC Comparison All Models #######

def plot_auc_comparison_all_models(model_results, save_path, metric_type="roc", n_classes=None):
    """
    Cria gráfico de barras comparando AUC (ROC ou PR) de todos os modelos.

    Parâmetros:
        model_results (dict): dicionário com resultados dos modelos. 
                              Exemplo:
                                {
                                  "modelo1": {"predictions": {"y_true": ..., "y_pred_proba": ...}},
                                  ...
                                }
        save_path (str): caminho base para salvar o gráfico (sem extensão)
        metric_type (str): "roc" para AUC-ROC ou "pr" para AUC-PR
        n_classes (int, opcional): número de classes esperadas (usado para validação)
    """
    assert metric_type in ["roc", "pr"], "metric_type deve ser 'roc' ou 'pr'."

    model_names = []
    auc_values = []

    for model_name, data in sorted(model_results.items()):
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])

            # Converte one-hot para rótulos, se necessário
            if y_true.ndim > 1:
                y_true = np.argmax(y_true, axis=1)

            if n_classes and y_pred_proba.shape[1] != n_classes:
                raise ValueError(f"Esperado {n_classes} classes, mas recebido {y_pred_proba.shape[1]}")

            # Escolhe métrica
            if metric_type == "roc":
                auc_value = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            else:  # metric_type == "pr"
                auc_value = average_precision_score(y_true, y_pred_proba, average='weighted')

            model_names.append(format_model_name(model_name))
            auc_values.append(auc_value)

        except Exception as e:
            print(f"⚠️ Erro processando {model_name}: {e}")
            continue

    if not model_names:
        print("❌ Nenhum modelo válido encontrado.")
        return

    # --- Plotagem ---
    plt.figure(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#17becf'] * 3  # repete se houver muitos modelos

    bars = plt.bar(range(len(model_names)), auc_values, 
                   color=colors[:len(model_names)], alpha=0.8,
                   edgecolor='black', linewidth=1)

    # Adiciona valores no topo das barras
    for i, (bar, auc) in enumerate(zip(bars, auc_values)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                 f'{auc:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

    # Título e eixos dinâmicos
    metric_label = "ROC-AUC" if metric_type == "roc" else "PR-AUC"
    plt.xlabel('Modelos', fontsize=22)
    plt.ylabel(f'{metric_label} (weighted)', fontsize=22)
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')

    # Linha de referência
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Classificador Aleatório')

    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()

    # Salvar gráficos
    plt.savefig(f"{save_path}_{metric_type}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}_{metric_type}.pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ {metric_label} comparison salvo em: {save_path}_{metric_type}.png / .pdf")

####### ROC OvR and OvO by model #######

def plot_ovr_vs_ovo_weighted_comparison(model_results, save_path, n_classes):
    """
    Plota comparação OvR vs OvO usando apenas weighted average
    """
    models_names = []
    ovr_weighted_scores = []
    ovo_weighted_scores = []
    
    for model_name, data in sorted(model_results.items()):
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])
            
            ovr_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            ovo_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='weighted')
            
            models_names.append(format_model_name(model_name))
            ovr_weighted_scores.append(ovr_weighted)
            ovo_weighted_scores.append(ovo_weighted)
        except Exception as e:
            continue
    
    plt.figure(figsize=(10, 8))
    
    x = np.arange(len(models_names))
    width = 0.35
    
    plt.bar(x - width/2, ovr_weighted_scores, width, label='OvR Weighted AUC', alpha=0.8)
    plt.bar(x + width/2, ovo_weighted_scores, width, label='OvO Weighted AUC', alpha=0.8)
    
    plt.xlabel('Models', fontsize=22)
    plt.ylabel('Weighted ROC-AUC Score', fontsize=22)
    plt.xticks(x, models_names, rotation=45, ha='right')
    plt.legend(fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()

def compute_ovr_roc_metrics(y_true, y_pred_proba, n_classes):
    """
    Computa métricas ROC usando esquema One-vs-Rest (OvR)
    Baseado em: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas [n_samples, n_classes]
        n_classes: Número de classes
        
    Returns:
        dict: Métricas ROC para cada classe e agregadas
    """
    # Binarizar labels usando One-vs-Rest
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Se temos apenas 2 classes, label_binarize retorna shape (n_samples, 1)
    # Precisamos expandir para (n_samples, 2)
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calcular ROC para cada classe (One-vs-Rest)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calcular micro-average ROC (agregar todas as classes)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Calcular macro-average ROC
    # Primeiro interpolar todas as curvas ROC em uma grade comum
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
    
    # Média das TPRs
    mean_tpr /= n_classes
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'y_true_bin': y_true_bin
    }

def compute_ovo_roc_metrics(y_true, y_pred_proba, n_classes, class_names):
    """
    Computa métricas ROC usando esquema One-vs-One (OvO)
    Baseado em: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas [n_samples, n_classes]
        n_classes: Número de classes
        class_names: Nomes das classes
        
    Returns:
        dict: Métricas ROC para cada par de classes
    """
    # Gerar todas as combinações possíveis de pares de classes
    class_pairs = list(combinations(range(n_classes), 2))
    
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    pair_scores = []
    pair_metrics = {}
    
    for class_a, class_b in class_pairs:
        # Criar máscaras para amostras das duas classes
        a_mask = y_true == class_a
        b_mask = y_true == class_b
        ab_mask = np.logical_or(a_mask, b_mask)
        
        if not np.any(ab_mask):
            continue
            
        # Extrair dados apenas dessas duas classes
        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]
        
        # Calcular ROC para classe A vs classe B
        fpr_a, tpr_a, _ = roc_curve(a_true, y_pred_proba[ab_mask, class_a])
        fpr_b, tpr_b, _ = roc_curve(b_true, y_pred_proba[ab_mask, class_b])
        
        # Interpolar em grade comum e calcular média
        mean_tpr = np.zeros_like(fpr_grid)
        mean_tpr += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr /= 2
        
        auc_score = auc(fpr_grid, mean_tpr)
        pair_scores.append(auc_score)
        
        pair_key = f"{class_names[class_a]}_vs_{class_names[class_b]}"
        pair_metrics[pair_key] = {
            'fpr': fpr_grid,
            'tpr': mean_tpr,
            'auc': auc_score,
            'fpr_a': fpr_a,
            'tpr_a': tpr_a,
            'auc_a': auc(fpr_a, tpr_a),
            'fpr_b': fpr_b,
            'tpr_b': tpr_b,
            'auc_b': auc(fpr_b, tpr_b),
            'class_a': class_a,
            'class_b': class_b
        }
    
    # Calcular macro-average de todos os pares
    macro_auc = np.mean(pair_scores) if pair_scores else 0.0
    
    return {
        'pairs': pair_metrics,
        'macro_auc': macro_auc,
        'pair_scores': pair_scores
    }

def plot_individual_model_ovr_roc(model_name, predictions, save_path, n_classes, class_names):
    """
    Plota ROC One-vs-Rest para um modelo individual
    Mostra curvas individuais: Classe 0 vs (1+2), Classe 1 vs (0+2), Classe 2 vs (0+1)
    """
    y_true = np.array(predictions['y_true'])
    y_pred_proba = np.array(predictions['y_pred_proba'])
    
    ovr_metrics = compute_ovr_roc_metrics(y_true, y_pred_proba, n_classes)
    
    plt.figure(figsize=(10, 8))
    
    # Cores para as classes individuais
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, Laranja, Verde
    
    # Plot curvas individuais para cada classe vs resto
    for i in range(n_classes):
        class_label = class_names[i] if i < len(class_names) else f'Class {i}'
        plt.plot(ovr_metrics['fpr'][i], ovr_metrics['tpr'][i], 
                color=colors[i], lw=3,
                label=f'{class_label} (AUC = {ovr_metrics["roc_auc"][i]:.3f})')
    
    # Weighted average usando sklearn
    weighted_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    
    # Plot weighted average line
    plt.axhline(y=0.5, color='gray', lw=1, linestyle=':', alpha=0.5)
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    model_display_name = format_model_name(model_name)
    plt.legend(loc="lower right", fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()

def plot_individual_model_ovo_roc(model_name, predictions, save_path, n_classes, class_names):
    """
    Plota ROC One-vs-One para um modelo individual
    """
    y_true = np.array(predictions['y_true'])
    y_pred_proba = np.array(predictions['y_pred_proba'])
    
    ovo_metrics = compute_ovo_roc_metrics(y_true, y_pred_proba, n_classes, class_names)
    
    plt.figure(figsize=(10, 8))
    
    # Plot pares individuais
    colors = cycle(MULTICLASS_COLORS)
    for pair_name, pair_metrics in ovo_metrics['pairs'].items():
        color = next(colors)
        pair_display = pair_name.replace('_', ' ')
        plt.plot(pair_metrics['fpr'], pair_metrics['tpr'], 
                color=color, lw=2,
                label=f'{pair_display} (AUC = {pair_metrics["auc"]:.3f})')
    
    # Macro-average OvO
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr_ovo = np.zeros_like(fpr_grid)
    
    for pair_metrics in ovo_metrics['pairs'].values():
        mean_tpr_ovo += pair_metrics['tpr']
    
    if len(ovo_metrics['pairs']) > 0:
        mean_tpr_ovo /= len(ovo_metrics['pairs'])
    
    plt.plot(fpr_grid, mean_tpr_ovo, color='red', lw=3, linestyle=':', 
            label=f'Macro-average (AUC = {ovo_metrics["macro_auc"]:.3f})')
    
    # Weighted average usando sklearn
    weighted_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='weighted')
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    model_display_name = format_model_name(model_name)
    plt.legend(loc="lower right", fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()

def plot_roc_ovr_individual_models(model_results, curves_dir, n_classes, class_names):
    """
    Cria gráficos ROC One-vs-Rest individuais para cada modelo
    Salva em pastas organizadas por nome do modelo dentro de curves/
    """
    for model_name, data in model_results.items():
        try:
            # Criar pasta para o modelo dentro de curves/
            model_curves_dir = os.path.join(curves_dir, model_name)
            os.makedirs(model_curves_dir, exist_ok=True)
            
            predictions = data['predictions']
            save_path = os.path.join(model_curves_dir, f"{model_name}_roc_ovr")
            plot_individual_model_ovr_roc(model_name, predictions, save_path, n_classes, class_names)
            print(f"ROC OvR salvo para {model_name} em {model_curves_dir}")
        except Exception as e:
            print(f"Erro gerando ROC OvR para {model_name}: {e}")

def plot_roc_ovo_individual_models(model_results, curves_dir, n_classes, class_names):
    """
    Cria gráficos ROC One-vs-One individuais para cada modelo
    Salva em pastas organizadas por nome do modelo dentro de curves/
    """
    for model_name, data in model_results.items():
        try:
            # Criar pasta para o modelo dentro de curves/
            model_curves_dir = os.path.join(curves_dir, model_name)
            os.makedirs(model_curves_dir, exist_ok=True)
            
            predictions = data['predictions']
            save_path = os.path.join(model_curves_dir, f"{model_name}_roc_ovo")
            plot_individual_model_ovo_roc(model_name, predictions, save_path, n_classes, class_names)
            print(f"ROC OvO salvo para {model_name} em {model_curves_dir}")
        except Exception as e:
            print(f"Erro gerando ROC OvO para {model_name}: {e}")


def plot_average_curves_all_models(model_results, curves_dir, average_type='macro', curve_type='roc', n_classes=3, class_names=None):
    """
    Gera um plot com curvas de todos os modelos usando average (micro ou macro).
    Permite escolher se quer ROC ou PR curves por parâmetro.
    
    Args:
        model_results (dict): Resultados dos modelos com predições
        curves_dir (str): Diretório para salvar os gráficos
        average_type (str): 'micro' ou 'macro' average
        curve_type (str): 'roc' ou 'pr' - tipo de curva
        n_classes (int): Número de classes
        class_names (list): Nomes das classes
        
    Returns:
        None: Salva os gráficos em arquivo
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    plt.figure(figsize=(12, 10))
    
    models_plotted = 0
    models_with_errors = []
    
    # Ordenar modelos para plotagem consistente
    sorted_models = sorted(model_results.items())
    
    for model_name, data in sorted_models:
        print(f"Processando {model_name} para {curve_type.upper()} {average_type} average...")
        
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred_proba = np.array(predictions['y_pred_proba'])
            
            # Verificar se os dados são consistentes
            if len(y_pred_proba.shape) == 1:
                print(f"  ⚠️ {model_name}: Dados binários detectados, pulando...")
                continue
                
            if y_pred_proba.shape[1] != n_classes:
                print(f"  ⚠️ {model_name}: Número de classes inconsistente ({y_pred_proba.shape[1]} vs {n_classes})")
                continue
            
            # Binarizar labels para multiclasse
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            if n_classes == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            
            model_display_name = format_model_name(model_name)
            color = STANDARD_COLORS.get(model_name.lower().replace('metrics', '').replace('_', '').replace('-', ''), f'C{models_plotted}')
            
            if curve_type == 'roc':
                if average_type == 'micro':
                    # Micro-average ROC
                    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
                    roc_auc_micro = auc(fpr_micro, tpr_micro)
                    
                    plt.plot(fpr_micro, tpr_micro, color=color, lw=3,
                            label=f'{model_display_name} (Micro AUC = {roc_auc_micro:.3f})')
                    
                elif average_type == 'macro':
                    # Macro-average ROC
                    fpr_grid = np.linspace(0.0, 1.0, 1000)
                    mean_tpr = np.zeros_like(fpr_grid)
                    
                    aucs = []
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                        aucs.append(auc(fpr, tpr))
                        mean_tpr += np.interp(fpr_grid, fpr, tpr)
                    
                    mean_tpr /= n_classes
                    macro_auc = np.mean(aucs)
                    
                    plt.plot(fpr_grid, mean_tpr, color=color, lw=3,
                            label=f'{model_display_name} (AUC = {macro_auc:.3f})')
                            
            elif curve_type == 'pr':
                if average_type == 'micro':
                    # Micro-average PR
                    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_proba.ravel())
                    pr_auc_micro = auc(recall_micro, precision_micro)
                    
                    plt.plot(recall_micro, precision_micro, color=color, lw=3,
                            label=f'{model_display_name} (Micro AUC = {pr_auc_micro:.3f})')
                    
                elif average_type == 'macro':
                    # Macro-average PR
                    recall_grid = np.linspace(0.0, 1.0, 1000)
                    mean_precision = np.zeros_like(recall_grid)
                    
                    aucs = []
                    for i in range(n_classes):
                        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                        aucs.append(auc(recall, precision))
                        
                        # Interpolar precision no grid de recall
                        # Inverter para interpolação correta (recall deve ser crescente)
                        precision_rev = precision[::-1]
                        recall_rev = recall[::-1]
                        mean_precision += np.interp(recall_grid, recall_rev, precision_rev)
                    
                    mean_precision /= n_classes
                    macro_auc = np.mean(aucs)
                    
                    plt.plot(recall_grid, mean_precision, color=color, lw=3,
                            label=f'{model_display_name} (AUC = {macro_auc:.3f})')
            
            models_plotted += 1
            print(f"  ✓ {model_name} processado com sucesso")
            
        except Exception as e:
            models_with_errors.append(model_name)
            print(f"  ❌ {model_name}: Erro ao processar - {e}")
            continue
    
    if models_plotted == 0:
        print("❌ Nenhum modelo foi plotado com sucesso!")
        plt.close()
        return
    
    # Configurações do gráfico baseadas no tipo de curva
    if curve_type == 'roc':
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=22)
        plt.ylabel('True Positive Rate', fontsize=22)
        
    elif curve_type == 'pr':
        # Baseline para PR (proporção da classe positiva)
        pos_rate = np.mean(y_true_bin)  # Média das classes positivas
        plt.axhline(y=pos_rate, color='gray', lw=2, linestyle='--', alpha=0.8,
                    label=f'Random Classifier (AP = {pos_rate:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=22)
        plt.ylabel('Precision', fontsize=22)
    
    plt.legend(loc="best", fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar gráficos
    filename_base = f"average_curves_all_models_{curve_type}_{average_type}"
    save_path = os.path.join(curves_dir, filename_base)
    
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Curvas {curve_type.upper()} {average_type} salvas em: {save_path}")
    
    # Relatório final
    if models_with_errors:
        print(f"⚠️  Modelos com erro: {', '.join(models_with_errors)}")


##########################################

##### MULTICLASS GENERATE PLOTS ######

def generate_multiclass_plots(model_results, class_names, experiment_dir, n_classes=3):
    """
    Função principal para gerar todos os gráficos relacionados a classificação multiclasse
    Inclui ROC Curves (OvR e OvO) e Precision-Recall Curves
    """

    sample_model = next(iter(model_results.values()))
    y_true_sample = np.array(sample_model['predictions']['y_true'])

    
    # Gerar gráficos ROC OvR individuais
    print("Gerando ROC OvR individuais...")
    roc_ovr_path = os.path.join(experiment_dir, "roc_ovr_curves")
    os.makedirs(roc_ovr_path, exist_ok=True)
    plot_roc_ovr_individual_models(model_results, roc_ovr_path, n_classes, class_names)
    
    # Gerar gráficos ROC OvO individuais
    print("Gerando ROC OvO individuais...")
    roc_ovo_path = os.path.join(experiment_dir, "roc_ovo_curves")
    os.makedirs(roc_ovo_path, exist_ok=True)
    plot_roc_ovo_individual_models(model_results, roc_ovo_path, n_classes, class_names)
    
    # Gerar gráficos PR individuais para cada modelo
    print("Gerando PR curves individuais para cada modelo...")
    pr_individual_path = os.path.join(experiment_dir, "pr_curves")
    os.makedirs(pr_individual_path, exist_ok=True)
    plot_pr_individual_models(model_results, pr_individual_path, n_classes, class_names)
    
    # Gerar comparação weighted AUC para todos os modelos (ROC)
    print("Gerando comparação weighted AUC (ROC) para todos os modelos...")
    weighted_auc_path = os.path.join(experiment_dir, "weighted_auc_comparison")
    os.makedirs(weighted_auc_path, exist_ok=True)
    weighted_auc_file_base = os.path.join(weighted_auc_path, "weighted_auc_comparison")
    plot_auc_comparison_all_models(model_results, weighted_auc_file_base, metric_type="roc", n_classes=n_classes)
    
    # Gerar comparação weighted AUC para todos os modelos (PR)
    print("Gerando comparação weighted AUC (PR) para todos os modelos...")
    plot_auc_comparison_all_models(model_results, weighted_auc_file_base, metric_type="pr", n_classes=n_classes)
    
    # Gerar curvas average de todos os modelos
    print("Gerando curvas average para todos os modelos...")
    average_curves_path = os.path.join(experiment_dir, "average_curves")
    os.makedirs(average_curves_path, exist_ok=True)
    
    # ROC Macro Average
    print("  - ROC Macro Average...")
    plot_average_curves_all_models(model_results, average_curves_path, 
                                 average_type='macro', curve_type='roc', 
                                 n_classes=n_classes, class_names=class_names)
    
    # PR Macro Average
    print("  - PR Macro Average...")
    plot_average_curves_all_models(model_results, average_curves_path, 
                                 average_type='macro', curve_type='pr', 
                                 n_classes=n_classes, class_names=class_names)
    
    # Gerar matrizes de confusão
    print("\nGerando Confusion Matrices...")
    cm_path = os.path.join(experiment_dir, "confusion_matrices")
    os.makedirs(cm_path, exist_ok=True)
    plot_confusion_matrices_multiclass(model_results, cm_path, class_names)

    
    print("✅ Todos os gráficos multiclasse foram gerados com sucesso!")


def plot_confusion_matrices_multiclass(model_results, save_dir, class_names):
    """
    Cria gráficos de matriz de confusão para todos os modelos multiclasse
    
    Args:
        model_results (dict): Resultados dos modelos com predições
        save_dir (str): Diretório para salvar os gráficos
        class_names (list): Nomes das classes
    """
    models_plotted = 0
    models_with_errors = []
    
    # If class_names not provided or generic, use proper labels
    if class_names is None or (len(class_names) > 0 and 'Class' in class_names[0]):
        class_names = ['Passenger', 'TSG', 'Oncogenes']
    
    for model_name, data in sorted(model_results.items()):
        print(f"  Processando matriz de confusão para {model_name}...")
        
        try:
            predictions = data['predictions']
            y_true = np.array(predictions['y_true'])
            y_pred = np.array(predictions.get('y_pred', []))
            
            # Se não tiver y_pred, usar y_pred_proba para gerar predições
            if len(y_pred) == 0:
                y_pred_proba = np.array(predictions['y_pred_proba'])
                y_pred = np.argmax(y_pred_proba, axis=1)
            
            model_display_name = format_model_name(model_name)
            
            # Calcular matriz de confusão
            cm = confusion_matrix(y_true, y_pred)
            
            # Criar figura maior para multiclasse
            plt.figure(figsize=(12, 10))
            
            # Usar seaborn para plotar heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 22})
            
            plt.ylabel('True Label', fontsize=22)
            plt.xlabel('Predicted Label', fontsize=22)
            plt.tight_layout()
            
            model_save_path = os.path.join(save_dir, f'cm_{model_name}')
            plt.savefig(model_save_path + '.png', dpi=300, bbox_inches='tight')
            plt.savefig(model_save_path + '.pdf', bbox_inches='tight')
            plt.close()
            
            models_plotted += 1
            
        except Exception as e:
            models_with_errors.append(model_name)
            print(f"    ❌ Erro: {e}")
    
    print(f"\n  Matrizes de confusão plotadas: {models_plotted}")
    if models_with_errors:
        print(f"  Modelos com erro: {', '.join(models_with_errors)}")


if __name__ == "__main__":
    """
    Permite executar o script diretamente para gerar plots multiclasse
    """
    # Import functions from plot_curves.py for standalone execution
    from plot_curves import browse_directories, load_from_forplots_directory, detect_classification_type
    
    print("\n" + "="*70, flush=True)
    print("GERAÇÃO DE GRÁFICOS MULTICLASSE", flush=True)
    print("="*70 + "\n", flush=True)
    
    print("Navegue até o diretório com os arquivos JSON dos modelos", flush=True)
    selected_dir = browse_directories(base_dir="./results_final_tcc")
    
    if not selected_dir:
        print("❌ Nenhum diretório selecionado. Encerrando.", flush=True)
        sys.exit(1)
    
    print(f"\n📂 Carregando dados de: {selected_dir}", flush=True)
    model_results = load_from_forplots_directory(selected_dir)
    
    if not model_results:
        print("❌ Nenhum modelo encontrado!", flush=True)
        sys.exit(1)
    
    # Detectar tipo de classificação
    first_model = next(iter(model_results.values()))
    predictions = first_model['predictions']
    y_true = np.array(predictions['y_true'])
    y_pred_proba = np.array(predictions['y_pred_proba'])
    classification_type, n_classes, class_names = detect_classification_type(y_true, y_pred_proba)
    
    print(f"✅ {len(model_results)} modelos carregados", flush=True)
    print(f"📊 Tipo: {classification_type} ({n_classes} classes)", flush=True)
    
    if classification_type != 'multiclass' or n_classes <= 2:
        print("⚠️  Este script é para classificação multiclasse (>2 classes)", flush=True)
        print(f"   Detectado: {classification_type} com {n_classes} classes", flush=True)
        sys.exit(1)
    
    # Criar diretório de saída
    curves_dir = os.path.join(selected_dir, "curves_multiclass")
    os.makedirs(curves_dir, exist_ok=True)
    
    print(f"\n🎨 Gerando gráficos multiclasse...", flush=True)
    generate_multiclass_plots(model_results, class_names, curves_dir, n_classes)
    
    print(f"\n💾 Gráficos salvos em: {curves_dir}", flush=True)
    print("\n" + "="*70, flush=True)
    print("PROCESSO FINALIZADO", flush=True)
    print("="*70 + "\n", flush=True)
