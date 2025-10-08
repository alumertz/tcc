#!/usr/bin/env python3
"""
Script principal para experimentação com modelos de classificação
para predição de genes-alvo usando dados ômicos.
"""

import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from processing import prepare_dataset, get_dataset_info
from models import (
    optimize_decision_tree_classifier,
    optimize_random_forest_classifier,
    optimize_gradient_boosting_classifier,
    optimize_hist_gradient_boosting_classifier,
    optimize_knn_classifier,
    optimize_mlp_classifier,
    optimize_svc_classifier
)

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
    ("Decision Tree", optimize_decision_tree_classifier),
    ("Random Forest", optimize_random_forest_classifier),
    ("Gradient Boosting", optimize_gradient_boosting_classifier),
    ("Histogram Gradient Boosting", optimize_hist_gradient_boosting_classifier),
    ("K-Nearest Neighbors", optimize_knn_classifier),
    ("Multi-Layer Perceptron", optimize_mlp_classifier),
    #("Support Vector Classifier", optimize_svc_classifier),
]

# Caminhos dos dados
FEATURES_PATH = "renan/data_files/omics_features/UNION_features.tsv"
LABELS_PATH = "renan/data_files/labels/UNION_labels.tsv"
RESULTS_PATH = "artigo/results/"


def select_features(X, selected_groups):
    """Seleciona colunas de X com base nos grupos de features"""
    indices = []
    for group in selected_groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Grupo inválido: {group}")
        indices.extend(FEATURE_GROUPS[group])

    return X[:, sorted(set(indices))]


def run_single_model(model_name, optimizer_func, X, y, n_trials):
    """Executa um único modelo de classificação"""
    print("=" * 80)
    print(f"EXECUTANDO MODELO: {model_name}")
    print("=" * 80)

    try:
        best_model = optimizer_func(X, y, n_trials=n_trials, save_results=True)
        print(f"✓ {model_name} executado com sucesso!")
        return {'model_name': model_name, 'status': 'success', 'model': best_model}

    except Exception as e:
        print(f"✗ Erro ao executar {model_name}: {e}")
        return {'model_name': model_name, 'status': 'error', 'error': str(e), 'model': None}


def run_all_models(X, y, n_trials):
    """Executa todos os modelos definidos"""
    print("\nINICIANDO EXPERIMENTAÇÃO COM TODOS OS MODELOS")
    print(f"Dataset: {X.shape[0]} amostras x {X.shape[1]} features")
    print(f"Trials por modelo: {n_trials}\n")

    results = []
    for i, (model_name, optimizer_func) in enumerate(MODELS_CONFIG, 1):
        print(f"Progresso: {i}/{len(MODELS_CONFIG)} modelos")
        results.append(run_single_model(model_name, optimizer_func, X, y, n_trials))
        time.sleep(2)  # pausa breve entre execuções
    return results


def summarize_results(results, selected_groups=None):
    """Cria um resumo textual dos resultados dos modelos"""
    if isinstance(results, dict):
        results = [results]

    if selected_groups:
        print(f"\nÔMICAS UTILIZADAS NO EXPERIMENTO: {', '.join(selected_groups)}\n")

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
    selected_groups = ["CNA"]
    X = select_features(X, selected_groups)

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
    N_TRIALS = 20
    print("\nCONFIGURAÇÃO DO EXPERIMENTO:")
    print(f"  Trials por modelo: {N_TRIALS}")
    print("  Validação: 5-fold estratificada + Holdout 80/20")
    print("  Métrica: PR AUC (Average Precision)")
    print(f"  Resultados: {RESULTS_PATH}\n")

    # === Escolha entre rodar um ou todos os modelos ===
    # Uncomment a linha desejada abaixo:

    # results = run_single_model("Gradient Boosting", optimize_gradient_boosting_classifier, X, y, N_TRIALS)
    # results = run_single_model("Support Vector Classifier", optimize_svc_classifier, X, y, N_TRIALS)
    # results = run_single_model("Multi-Layer Perceptron", optimize_mlp_classifier, X, y, N_TRIALS)
    results = run_all_models(X, y, N_TRIALS)
    #results = run_single_model("K-Nearest Neighbors", optimize_knn_classifier, X, y, N_TRIALS)
    summarize_results(results, selected_groups)
    print("\nEXPERIMENTO CONCLUÍDO!")
    print(f"Resultados salvos em: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
