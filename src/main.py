#!/usr/bin/env python3
"""
Script principal para experimentação com modelos de classificação
para predição de genes-alvo usando dados ômicos.
"""

import os
import time
import warnings
import numpy as np
import argparse
import itertools

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
        #"n_estimators": 200,
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


def get_data_paths(use_renan=False):
    """
    Retorna os caminhos dos arquivos baseado na fonte de dados escolhida
    
    Args:
        use_renan (bool): Se True, usa arquivos do Renan; se False, usa arquivos da Ana
        
    Returns:
        tuple: (features_path, labels_path, data_source)
    """
    if use_renan:
        features_path = "renan/data_files/omics_features/UNION_features.tsv"
        labels_path = "renan/data_files/labels/UNION_labels.tsv"
        data_source = "RENAN"
        print(f"Usando dados do RENAN:")
        print(f"   Features: renan/data_files/omics_features/UNION_features.tsv")
        print(f"   Labels: renan/data_files/labels/UNION_labels.tsv")
        print(f"   Formato labels: gene, label (True/False/NaN)")
    else:
        features_path = "data/UNION_features.tsv"
        labels_path = "data/processed/UNION_labels.tsv"
        data_source = "ANA"
        print(f"Usando dados da ANA:")
        print(f"   Features: data/UNION_features.tsv")
        print(f"   Labels: data/processed/UNION_labels.tsv") 
        print(f"   Formato labels: genes, 2class (binary), 3class (multiclass)")
    print()
    
    return features_path, labels_path, data_source


def parse_arguments():
    """
    Processa argumentos da linha de comando
    
    Returns:
        argparse.Namespace: Argumentos processados
    """
    parser = argparse.ArgumentParser(
        description='Experimentação com modelos de classificação para predição de genes-alvo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py                    # Usa arquivos da Ana, classificação binária (padrão)
  python main.py -renan             # Usa arquivos do Renan (formato original)
  python main.py -multiclass        # Usa classificação multiclasse (TSG vs Oncogene vs Passenger)
  python main.py -multiclass -renan # Combina ambas opções
  python main.py --help             # Mostra esta ajuda
        """
    )
    
    parser.add_argument(
        '-renan', '--renan', 
        action='store_true',
        help='Usa arquivos de dados do Renan (formato original: gene, label)'
    )
    
    parser.add_argument(
        '-multiclass', '--multiclass',
        action='store_true',
        help='Usa classificação multiclasse (TSG=1, Oncogene=2) ao invés de binária (cancer=1)'
    )
    
    return parser.parse_args()


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
        best_model = optimizer_func(X, y, params, save_results=True, omics_used=omics_used)

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


'''def main(use_renan=False, use_multiclass=False):
    """
    Função principal do experimento
    
    Args:
        use_renan (bool): Se True, usa arquivos do Renan; se False, usa arquivos da Ana
        use_multiclass (bool): Se True, usa classificação multiclasse; se False, usa binária
    """
    print("CLASSIFICAÇÃO DE GENES-ALVO USANDO DADOS ÔMICOS")
    print("="*80)
    
    # Obtém os caminhos dos arquivos baseado na fonte escolhida
    features_path, labels_path, data_source = get_data_paths(use_renan)
    
    # Verifica se os arquivos existem
    if not os.path.exists(features_path):
        print(f"❌ Arquivo de features não encontrado: {features_path}")
        return
    
    if not os.path.exists(labels_path):
        print(f"❌ Arquivo de labels não encontrado: {labels_path}")
        return
    
    print("✅ Arquivos encontrados com sucesso!")
    
    # Prepara o dataset
    print("🔄 Carregando e preparando dados...")
    classification_type = 'multiclass' if use_multiclass else 'binary'
    X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path, classification_type)


    # Seleção de features
    # "CNA", "Gene_Expression", "DNA_Methylation", "Mutations"
    omics_used = ["CNA"]
    X = select_features(X, omics_used)

    if X is None:
        print("❌ Erro ao preparar dataset. Abortando.")
        return
    
    # Mostra informações do dataset
    dataset_info = get_dataset_info(X, y, gene_names, feature_names)
    print("\n📊 INFORMAÇÕES DO DATASET:")
    print(f"  Fonte de dados: {data_source}")
    print(f"  Amostras: {dataset_info['n_samples']}")
    print(f"  Features: {dataset_info['n_features']}")
    print(f"  Distribuição das classes: {dataset_info['class_distribution']}")
    print(f"  Estatísticas das features:")
    print(f"    - Média: {dataset_info['feature_stats']['mean']:.4f}")
    print(f"    - Desvio padrão: {dataset_info['feature_stats']['std']:.4f}")
    print(f"    - Valores zero: {dataset_info['feature_stats']['zeros_percentage']:.2f}%")
    
    
    print(f"\n⚙️  CONFIGURAÇÃO DO EXPERIMENTO:")
    print(f"  Validação: Estratificada 5-fold + Holdout 80/20")
    print(f"  Métrica de otimização: PR AUC (Average Precision)")
    print(f"  Resultados salvos em: /Users/i583975/git/tcc/results/")
    print()
    
    # Executa todos os modelos
    #results = run_all_models(X, y, )
    #results = run_single_model("Decision Tree", train_decision_tree_classifier, X, y, omics_used)
    #results = run_single_model("Random Forest", train_random_forest_classifier, X, y, omics_used)
    #results = run_single_model("Gradient Boosting", train_gradient_boosting_classifier, X, y, omics_used)
    #results = run_single_model("K Nearest Neighbors", train_knn_classifier, X, y, omics_used)
    #results = run_single_model("Histogram Gradient Boosting,", train_hist_gradient_boosting_classifier, X, y, omics_used)
    #results = run_single_model("Multi-Layer Perceptron", train_mlp_classifier, X, y, omics_used)
    #results = run_single_model("Support Vector Classifier", train_svc_classifier, X, y, omics_used)
    results = run_single_model("CatBoost", train_catboost_classifier, X, y, omics_used)

    # Resumo final
    summarize_results(results)
    
    print("\n🎉 EXPERIMENTO CONCLUÍDO!")
    print("💾 Resultados salvos em arquivos organizados por modelo.")


if __name__ == "__main__":
    # Processa argumentos da linha de comando
    args = parse_arguments()
    
    # Executa o experimento com a fonte de dados escolhida
    main(args.renan, args.multiclass)'''


def run_all_omics_combinations(use_renan=False, use_multiclass=False): # Roda combinações de 1 e 3
    all_omics = ["CNA", "Gene_Expression", "DNA_Methylation", "Mutations"]

    print("🧪 Executando com cada omic individualmente...")
    for omic in all_omics:
        print(f"\n🔬 Rodando experimento com: [{omic}]")
        main(use_renan=use_renan, use_multiclass=use_multiclass, omics_used=[omic])

    print("\n🧪 Executando com combinações de 3 omics (total - 1)...")
    comb_3 = list(itertools.combinations(all_omics, 3))
    for combo in comb_3:
        print(f"\n🔬 Rodando experimento com: {list(combo)}")
        main(use_renan=use_renan, use_multiclass=use_multiclass, omics_used=list(combo))


def main(use_renan=False, use_multiclass=False, omics_used=None):    
    """
    Função principal do experimento
    
    Args:
        use_renan (bool): Se True, usa arquivos do Renan; se False, usa arquivos da Ana
        use_multiclass (bool): Se True, usa classificação multiclasse; se False, usa binária
    """
    print("CLASSIFICAÇÃO DE GENES-ALVO USANDO DADOS ÔMICOS")
    print("="*80)
    
    # Obtém os caminhos dos arquivos baseado na fonte escolhida
    features_path, labels_path, data_source = get_data_paths(use_renan)
    
    # Verifica se os arquivos existem
    if not os.path.exists(features_path):
        print(f"❌ Arquivo de features não encontrado: {features_path}")
        return
    
    if not os.path.exists(labels_path):
        print(f"❌ Arquivo de labels não encontrado: {labels_path}")
        return
    
    print("✅ Arquivos encontrados com sucesso!")
    
    # Prepara o dataset
    print("🔄 Carregando e preparando dados...")
    classification_type = 'multiclass' if use_multiclass else 'binary'
    X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path, classification_type)


    # Seleção de features
    # "CNA", "Gene_Expression", "DNA_Methylation", "Mutations"
    # omics_used = ["CNA", "Gene_Expression", "DNA_Methylation", "Mutations"]
    X = select_features(X, omics_used)

    if X is None:
        print("❌ Erro ao preparar dataset. Abortando.")
        return
    
    # Mostra informações do dataset
    dataset_info = get_dataset_info(X, y, gene_names, feature_names)
    print("\n📊 INFORMAÇÕES DO DATASET:")
    print(f"  Fonte de dados: {data_source}")
    print(f"  Amostras: {dataset_info['n_samples']}")
    print(f"  Features: {dataset_info['n_features']}")
    print(f"  Distribuição das classes: {dataset_info['class_distribution']}")
    print(f"  Estatísticas das features:")
    print(f"    - Média: {dataset_info['feature_stats']['mean']:.4f}")
    print(f"    - Desvio padrão: {dataset_info['feature_stats']['std']:.4f}")
    print(f"    - Valores zero: {dataset_info['feature_stats']['zeros_percentage']:.2f}%")
    
    
    print(f"\n⚙️  CONFIGURAÇÃO DO EXPERIMENTO:")
    print(f"  Validação: Estratificada 5-fold + Holdout 80/20")
    print(f"  Métrica de otimização: PR AUC (Average Precision)")
    print(f"  Resultados salvos em: /Users/i583975/git/tcc/results/")
    print()
    
    # Executa todos os modelos
    results = run_all_models(X, y, omics_used)
    #results = run_single_model("Decision Tree", train_decision_tree_classifier, X, y, omics_used)

    # Resumo final
    summarize_results(results)
    
    print("\n🎉 EXPERIMENTO CONCLUÍDO!")
    print("💾 Resultados salvos em arquivos organizados por modelo.")


if __name__ == "__main__":
    args = parse_arguments()
    #run_all_omics_combinations(use_renan=args.renan, use_multiclass=args.multiclass)

    # Rodando todos os modelos usando todas as ômicas
    all_omics = ["CNA", "Gene_Expression", "DNA_Methylation", "Mutations"]
    main(use_renan=args.renan, use_multiclass=args.multiclass, omics_used=all_omics)