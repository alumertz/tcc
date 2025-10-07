#!/usr/bin/env python3
"""
Arquivo principal para experimentação com modelos de classificação
para predição de genes-alvo usando dados ômicos.
"""

import sys
import os
import argparse
sys.path.append('/Users/i583975/git/tcc')

import numpy as np
import pandas as pd
from processing import prepare_dataset, get_dataset_info, split_dataset
from process_data import process_canonical, process_candidates
from models import (
    optimize_decision_tree_classifier,
    optimize_random_forest_classifier,
    optimize_gradient_boosting_classifier,
    optimize_hist_gradient_boosting_classifier,
    optimize_knn_classifier,
    optimize_mlp_classifier,
    optimize_svc_classifier,
    optimize_catboost_classifier
)
import warnings
warnings.filterwarnings('ignore')

def get_data_paths(use_renan=False):
    """
    Retorna os caminhos dos arquivos baseado na fonte de dados escolhida
    
    Args:
        use_renan (bool): Se True, usa arquivos do Renan; se False, usa arquivos da Ana
        
    Returns:
        tuple: (features_path, labels_path, data_source)
    """
    if use_renan:
        features_path = "/Users/i583975/git/tcc/renan/data_files/omics_features/UNION_features.tsv"
        labels_path = "/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv"
        data_source = "RENAN"
        print(f"📁 Usando dados do RENAN:")
        print(f"   Features: renan/data_files/omics_features/UNION_features.tsv")
        print(f"   Labels: renan/data_files/labels/UNION_labels.tsv")
        print(f"   Formato labels: gene, label (True/False/NaN)")
    else:
        features_path = "/Users/i583975/git/tcc/data/UNION_features.tsv"
        labels_path = "/Users/i583975/git/tcc/data/processed/UNION_labels.tsv"
        data_source = "ANA"
        print(f"📁 Usando dados da ANA:")
        print(f"   Features: data/UNION_features.tsv")
        print(f"   Labels: data/processed/UNION_labels.tsv") 
        print(f"   Formato labels: symbol, Oncogene, TSG, category")
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
  python main.py              # Usa arquivos da Ana (formato novo - padrão)
  python main.py -renan       # Usa arquivos do Renan (formato original)
  python main.py --help       # Mostra esta ajuda
        """
    )
    
    parser.add_argument(
        '-renan', '--renan', 
        action='store_true',
        help='Usa arquivos de dados do Renan (formato original: gene, label)'
    )
    
    return parser.parse_args()

def run_single_model(model_name, optimizer_func, X, y, n_trials=10):
    """
    Executa um único modelo de classificação
    
    Args:
        model_name (str): Nome do modelo
        optimizer_func (function): Função de otimização do modelo
        X (np.array): Features
        y (np.array): Labels
        n_trials (int): Número de trials para otimização
        
    Returns:
        dict: Resultados do modelo
    """
    print("="*80)
    print(f"EXECUTANDO MODELO: {model_name}")
    print("="*80)
    
    try:
        # Executa otimização (com salvamento automático)
        best_model = optimizer_func(X, y, n_trials=n_trials, save_results=True)
        
        results = {
            'model_name': model_name,
            'status': 'success',
            'model': best_model
        }
        
        print(f"✓ {model_name} executado com sucesso!")
        return results
        
    except Exception as e:
        print(f"✗ Erro ao executar {model_name}: {e}")
        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e),
            'model': None
        }


def run_all_models(X, y, n_trials=10):
    """
    Executa todos os modelos de classificação
    
    Args:
        X (np.array): Features
        y (np.array): Labels
        n_trials (int): Número de trials para otimização
        
    Returns:
        list: Lista com resultados de todos os modelos
    """
    # Definição dos modelos e suas funções de otimização
    models_config = [
        ("Decision Tree", optimize_decision_tree_classifier),
        ("Random Forest", optimize_random_forest_classifier),
        ("Gradient Boosting", optimize_gradient_boosting_classifier),
        ("Histogram Gradient Boosting", optimize_hist_gradient_boosting_classifier),
        ("K-Nearest Neighbors", optimize_knn_classifier),
        ("Multi-Layer Perceptron", optimize_mlp_classifier),
        ("Support Vector Classifier", optimize_svc_classifier),
        ("CatBoost", optimize_catboost_classifier)
    ]
    
    results = []
    
    print("INICIANDO EXPERIMENTAÇÃO COM TODOS OS MODELOS")
    print(f"Dataset: {X.shape[0]} amostras x {X.shape[1]} features")
    print(f"Número de trials por modelo: {n_trials}")
    print()
    
    for i, (model_name, optimizer_func) in enumerate(models_config, 1):
        print(f"Progresso: {i}/{len(models_config)} modelos")
        
        result = run_single_model(model_name, optimizer_func, X, y, n_trials)
        results.append(result)
        
        # Breve pausa entre modelos
        import time
        time.sleep(2)
    
    return results


def summarize_results(results):
    """
    Cria um resumo dos resultados de todos os modelos
    
    Args:
        results (list or dict): Lista com resultados dos modelos ou resultado único
    """
    print("="*80)
    print("RESUMO DOS RESULTADOS")
    print("="*80)
    
    # Se results for um único resultado (dict), converte para lista
    if isinstance(results, dict):
        results = [results]
    
    successful_models = [r for r in results if r['status'] == 'success']
    failed_models = [r for r in results if r['status'] == 'error']
    
    print(f"Modelos executados com sucesso: {len(successful_models)}")
    print(f"Modelos com erro: {len(failed_models)}")
    print()
    
    if successful_models:
        print("MODELOS BEM-SUCEDIDOS:")
        for result in successful_models:
            print(f"  • {result['model_name']}")
    
    if failed_models:
        print("\nMODELOS COM ERRO:")
        for result in failed_models:
            print(f"  • {result['model_name']}: {result['error']}")
    
    print("="*80)


def main(use_renan=False):
    """
    Função principal do experimento
    
    Args:
        use_renan (bool): Se True, usa arquivos do Renan; se False, usa arquivos da Ana
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
    X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path)
    
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
    
    # Configuração do experimento
    N_TRIALS = 30  # Número de trials por modelo (ajustar conforme necessário)
    
    print(f"\n⚙️  CONFIGURAÇÃO DO EXPERIMENTO:")
    print(f"  Trials por modelo: {N_TRIALS}")
    print(f"  Validação: Estratificada 5-fold + Holdout 80/20")
    print(f"  Métrica de otimização: PR AUC (Average Precision)")
    print(f"  Resultados salvos em: /Users/i583975/git/tcc/results/")
    print()
    
    # Executa todos os modelos
    print("🚀 Iniciando experimentos...")
    #results = run_all_models(X, y, n_trials=N_TRIALS)
    #results = run_single_model("Gradient Boosting", optimize_gradient_boosting_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("Decision Tree", optimize_decision_tree_classifier, X, y, n_trials=N_TRIALS)
    results = run_single_model("Support Vector Classifier", optimize_svc_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("Multi-Layer Perceptron", optimize_mlp_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("CatBoost", optimize_catboost_classifier, X, y, n_trials=N_TRIALS)

    # Resumo final
    summarize_results(results)
    
    print("\n🎉 EXPERIMENTO CONCLUÍDO!")
    print("💾 Resultados salvos em arquivos organizados por modelo.")


if __name__ == "__main__":
    # Processa argumentos da linha de comando
    args = parse_arguments()
    
    # Executa o experimento com a fonte de dados escolhida
    main(args.renan)
