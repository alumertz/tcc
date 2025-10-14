#!/usr/bin/env python3
"""
Script principal para experimentação com modelos de classificação
para predição de genes-alvo usando dados ômicos.
"""

import os
import time
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

from processing import prepare_dataset, get_dataset_info
from models import (
    train_decision_tree_classifier,
    train_random_forest_classifier,
    train_gradient_boosting_classifier,
    train_hist_gradient_boosting_classifier,
    train_knn_classifier,
    train_mlp_classifier,    
    train_svc_classifier,
    train_catboost_classifier
)

# Grupos de melhores hiperparâmetros
HYPERPARAMS = {
    "Decision Tree": {
        "max_depth": 3,
        "min_samples_split": 16,
        "min_samples_leaf": 1,
        "criterion": "gini"
    },
    "Random Forest": {
        "n_estimators": 200,
        "max_depth": 7,
        "min_samples_split": 11,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "criterion": "gini"
    },
    "Gradient Boosting": {
        "n_estimators": 150,
        "learning_rate": 0.041901086122355526,
        "max_depth": 4,
        "min_samples_split": 6,
        "min_samples_leaf": 8,
        "subsample": 0.8750118011622728,
    },
    "Histogram Gradient Boosting": {
        "max_iter": 68,
        "learning_rate": 0.07520558542833868,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "l2_regularization": 0.24633014373667755,
    },
    "K-Nearest Neighbors": {
        "n_neighbors": 20,
        "weights": "distance",
        "algorithm": "auto",
        "p": 2,
    },
    "Multi-Layer Perceptron": {
        "hidden_layer_sizes": (131, 115, 30),
        "activation": "logistic",
        "alpha": 0.07698392854033051,
        "learning_rate": "constant",
        "max_iter": 742,
    },
    "Support Vector Classifier": {
        "kernel": "rbf",
        "C": 0.3378198845315039,
        "probability": True,
        "max_iter": 1000,
        "tol": 0.001,
        "cache_size": 200,
        "gamma": "scale",
    },
    "CatBoost": {
        "iterations": 256,
        "learning_rate": 0.019261294694430553,
        "depth": 7,
        "l2_leaf_reg": 2.9603859610055303,
        "border_count": 239,
        "bagging_temperature": 0.30218907193827815,
        "random_strength": 0.8356422919240496,
        "verbose": False,
        "allow_writing_files": False,
    }
}


# Grupos de features por posição
FEATURE_GROUPS = {
    "CNA": list(range(0, 16)),
    "Gene_Expression": list(range(16, 32)),
    "DNA_Methylation": list(range(32, 48)),
    "Mutations": list(range(48, 64)),
    "Multiomics": list(range(0, 64)), 
}

# Configuração dos modelos e otimizadores
MODELS_CONFIG = [
    ("Decision Tree", train_decision_tree_classifier),
    ("Random Forest", train_random_forest_classifier),
    ("Gradient Boosting", train_gradient_boosting_classifier),
    ("Histogram Gradient Boosting", train_hist_gradient_boosting_classifier),
    ("K-Nearest Neighbors", train_knn_classifier),
    ("Multi-Layer Perceptron", train_mlp_classifier),
    ("Support Vector Classifier", train_svc_classifier),
    ("CatBoost", train_catboost_classifier)
]

# Caminhos dos dados
FEATURES_PATH = "renan/data_files/omics_features/UNION_features.tsv"
LABELS_PATH = "renan/data_files/labels/UNION_labels.tsv"
RESULTS_PATH = "results/omics"


def select_features(X, selected_groups):
    """Seleciona colunas de X com base nos grupos de features"""
    indices = []
    for group in selected_groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Grupo inválido: {group}")
        indices.extend(FEATURE_GROUPS[group])

    return X[:, sorted(set(indices))]


def run_single_model(model_name, optimizer_func, X, y, omics_used=None):
    print("=" * 80)
    print(f"EXECUTANDO MODELO: {model_name}")
    print("=" * 80)

    try:
        # Pega os hiperparâmetros para este modelo
        params = HYPERPARAMS.get(model_name, {})

        # Passa os hiperparâmetros dinamicamente via **kwargs
        best_model = optimizer_func(X, y, save_results=True, omics_used=omics_used, **params)

        print(f"✓ {model_name} executado com sucesso!")
        return {
            'model_name': model_name,
            'status': 'success',
            'model': best_model,
            'omics_used': omics_used if omics_used else []
        }

    except Exception as e:
        print(f"✗ Erro ao executar {model_name}: {e}")
        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e),
            'model': None,
            'omics_used': omics_used or []
        }


def run_all_models(X, y, omics_used=None):
    print("\nINICIANDO EXPERIMENTAÇÃO COM TODOS OS MODELOS")
    print(f"Dataset: {X.shape[0]} amostras x {X.shape[1]} features\n")

    results = []
    for i, (model_name, optimizer_func) in enumerate(MODELS_CONFIG, 1):
        print(f"Progresso: {i}/{len(MODELS_CONFIG)} modelos")
        results.append(run_single_model(model_name, optimizer_func, X, y, omics_used=omics_used))
        time.sleep(2)
    return results


def summarize_results(results, omics_used=None):
    """Cria um resumo textual dos resultados dos modelos"""
    if isinstance(results, dict):
        results = [results]

    if omics_used:
        print(f"\nÔMICAS UTILIZADAS NO EXPERIMENTO: {', '.join(omics_used)}\n")

    print("=" * 80)
    print("RESUMO DOS RESULTADOS")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']

    print(f"Modelos executados com sucesso: {len(successful)}")
    print(f"Modelos com erro: {len(failed)}\n")

    if successful:
        print("MODELOS BEM-SUCEDIDOS:")
        for r in successful:
            print(f"  • {r['model_name']}")

    if failed:
        print("\nMODELOS COM ERRO:")
        for r in failed:
            print(f"  • {r['model_name']}: {r['error']}")

    print("=" * 80)


def main():
    """Função principal"""
    print("\nCLASSIFICAÇÃO DE GENES-ALVO USANDO DADOS ÔMICOS")
    print("=" * 80)

    if not os.path.exists(FEATURES_PATH):
        print(f"Arquivo de features não encontrado: {FEATURES_PATH}")
        return
    if not os.path.exists(LABELS_PATH):
        print(f"Arquivo de labels não encontrado: {LABELS_PATH}")
        return

    print("Carregando e preparando dados...")
    X, y, gene_names, feature_names = prepare_dataset(FEATURES_PATH, LABELS_PATH)
    if X is None:
        print("Erro ao preparar dataset. Abortando.")
        return

    # Seleção de features
    # "CNA", "Gene_Expression", "DNA_Methylation", "Mutations"
    omics_used = ["CNA"]
    X = select_features(X, omics_used)

    # Informações gerais
    info = get_dataset_info(X, y, gene_names, feature_names)
    print("\nINFORMAÇÕES DO DATASET:")
    print(f"  Amostras: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Distribuição das classes: {info['class_distribution']}")
    print(f"  Estatísticas das features:")
    print(f"    - Média: {info['feature_stats']['mean']:.4f}")
    print(f"    - Desvio padrão: {info['feature_stats']['std']:.4f}")
    print(f"    - Valores zero: {info['feature_stats']['zeros_percentage']:.2f}%")

    # Configurações
    print("\nCONFIGURAÇÃO DO EXPERIMENTO:")
    print("  Validação: 5-fold estratificada + Holdout 80/20")
    print("  Métrica: PR AUC (Average Precision)")
    print(f"  Resultados: {RESULTS_PATH}\n")

    results = run_single_model("Decision Tree", train_decision_tree_classifier, X, y, omics_used)
    #results = run_all_models(X, y, omics_used)
    summarize_results(results, omics_used)
    print("\nEXPERIMENTO CONCLUÍDO!")
    print(f"Resultados salvos em: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
