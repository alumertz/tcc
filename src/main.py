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
from src.processing import prepare_dataset, get_dataset_info
from src.models import (
    optimize_decision_tree_classifier,
    optimize_random_forest_classifier,
    optimize_gradient_boosting_classifier,
    optimize_hist_gradient_boosting_classifier,
    optimize_knn_classifier,
    optimize_mlp_classifier,
    optimize_svc_classifier,
    optimize_catboost_classifier,
    optimize_xgboost_classifier
)
from src.reports import summarize_results
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from evaluation import evaluate_model_default
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
        features_path = "./renan/data_files/omics_features/UNION_features.tsv"
        labels_path = "./renan/data_files/labels/UNION_labels.tsv"
        data_source = "RENAN"
        print(f"📁 Usando dados do RENAN:")
        print(f"   Features: renan/data_files/omics_features/UNION_features.tsv")
        print(f"   Labels: renan/data_files/labels/UNION_labels.tsv")
        print(f"   Formato labels: gene, label (True/False/NaN)")
    else:
        features_path = "./data/UNION_features.tsv"
        labels_path = "./data/processed/UNION_labels.tsv"
        data_source = "ANA"
        print(f"📁 Usando dados da ANA:")
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
  python main.py                    # Modelos otimizados, dados Ana, classificação binária
  python main.py -default           # Modelos com parâmetros padrão (rápido)
  python main.py -renan             # Usa arquivos do Renan (formato original)
  python main.py -multiclass        # Classificação multiclasse (TSG vs Oncogene vs Passenger)
  python main.py -default -multiclass # Parâmetros padrão + multiclasse
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
    
    parser.add_argument(
        '-default', '--default',
        action='store_true',
        help='Executa modelos com parâmetros padrão (sem otimização Optuna)'
    )
    
    return parser.parse_args()

def run_single_model_optimize(model_name, optimizer_func, X, y, n_trials=10, data_source="ana", classification_type="binary"):
    """
    Executa um único modelo de classificação
    
    Args:
        model_name (str): Nome do modelo
        optimizer_func (function): Função de otimização do modelo
        X (np.array): Features
        y (np.array): Labels
        n_trials (int): Número de trials para otimização
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
        
    Returns:
        dict: Resultados do modelo
    """
    print("="*80)
    print(f"EXECUTANDO MODELO: {model_name}")
    print("="*80)
    
    try:
        # Executa otimização (com salvamento automático)
        best_model, test_metrics = optimizer_func(X, y, n_trials=n_trials, save_results=True, 
                                                data_source=data_source, classification_type=classification_type,
                                                use_nested_cv=True, outer_cv_folds=5)
        
        results = {
            'model_name': model_name,
            'status': 'success',
            'model': best_model,
            'test_metrics': test_metrics
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

def run_all_models_optimize(X, y, n_trials=10, data_source="ana", classification_type="binary"):
    """
    Executa todos os modelos de classificação
    
    Args:
        X (np.array): Features
        y (np.array): Labels
        n_trials (int): Número de trials para otimização
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
        
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
        ("CatBoost", optimize_catboost_classifier),
        ("XGBoost", optimize_xgboost_classifier)
    ]
    
    results = []
    
    print("INICIANDO EXPERIMENTAÇÃO COM TODOS OS MODELOS")
    print(f"Dataset: {X.shape[0]} amostras x {X.shape[1]} features")
    print(f"Número de trials por modelo: {n_trials}")
    print()
    
    for i, (model_name, optimizer_func) in enumerate(models_config, 1):
        print(f"Progresso: {i}/{len(models_config)} modelos")
        
        result = run_single_model_optimize(model_name, optimizer_func, X, y, n_trials, data_source, classification_type)
        results.append(result)
        
        # Breve pausa entre modelos
        import time
        time.sleep(2)
    
    return results

def run_all_default_models(X, y, data_source="ana", classification_type="binary"):
    """
    Executa todos os modelos com parâmetros padrão
    Pipeline unificado para todos: StandardScaler + Classifier
    
    Args:
        X (np.array): Features
        y (np.array): Labels
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
        
    Returns:
        list: Lista com resultados de todos os modelos
    """
    # Modelos com parâmetros padrão (mesma lista do main_default_models.py)
    default_models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ("Histogram Gradient Boosting", HistGradientBoostingClassifier(random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Multi-Layer Perceptron", MLPClassifier(random_state=42, max_iter=1000)),
        ("Support Vector Classifier", SVC(probability=True, random_state=42)),
        ("CatBoost", CatBoostClassifier(random_state=42, verbose=False)),
        ("XGBoost", XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss'))
    ]
    
    results = []
    
    print("INICIANDO AVALIAÇÃO COM PARÂMETROS PADRÃO")
    print(f"Dataset: {X.shape[0]} amostras x {X.shape[1]} features")
    print(f"Pipeline: StandardScaler + Classifier (unificado)")
    print(f"Métricas: Binary (precision, recall, f1)")
    print()
    
    for i, (model_name, model) in enumerate(default_models, 1):
        print(f"\nProgresso: {i}/{len(default_models)} modelos")
        
        try:
            result = evaluate_model_default(model, model_name, X, y, data_source, classification_type)
            results.append(result)
            print(f"✓ {model_name} executado com sucesso!")
            
        except Exception as e:
            print(f"✗ Erro ao executar {model_name}: {e}")
            results.append({
                'model_name': model_name,
                'status': 'error',
                'error': str(e)
            })
    
    return results

def main(use_renan=False, use_multiclass=False, use_default=False):
    """
    Função principal do experimento
    
    Args:
        use_renan (bool): Se True, usa arquivos do Renan; se False, usa arquivos da Ana
        use_multiclass (bool): Se True, usa classificação multiclasse; se False, usa binária
        use_default (bool): Se True, usa parâmetros padrão; se False, otimiza com Optuna
    """
    print("CLASSIFICAÇÃO DE GENES-ALVO USANDO DADOS ÔMICOS")
    print("="*80)
    
    # Inicializar timestamp do experimento para toda a sessão
    from src.reports import set_experiment_timestamp
    experiment_timestamp = set_experiment_timestamp()
    print(f"🕐 Timestamp do experimento: {experiment_timestamp}")
    print()
    
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
    
    # Configuração do experimento baseado no modo escolhido
    if use_default:
        print(f"\n⚙️  CONFIGURAÇÃO DO EXPERIMENTO (PARÂMETROS PADRÃO):")
        print(f"  Modo: Parâmetros padrão do scikit-learn")
        print(f"  Pipeline: StandardScaler + Classifier (unificado)")
        print(f"  Métricas: Binary (precision, recall, f1)")
        print(f"  Validação: Estratificada 5-fold + Holdout 80/20")
        print(f"  Tempo estimado: ~2-5 minutos")
        print(f"  Resultados salvos em: /Users/i583975/git/tcc/results/")
        print()
        
        # Executa modelos com parâmetros padrão
        print("🚀 Iniciando experimentos com parâmetros padrão...")
        data_source = "renan" if use_renan else "ana"
        classification_type = "multiclass" if use_multiclass else "binary"
        results = run_all_default_models(X, y, data_source, classification_type)
        
    else:
        N_TRIALS = 30  # Número de trials por modelo
        print(f"\n⚙️  CONFIGURAÇÃO DO EXPERIMENTO (OTIMIZAÇÃO):")
        print(f"  Modo: Otimização com Optuna")
        print(f"  Trials por modelo: {N_TRIALS}")
        print(f"  Pipeline: StandardScaler + Classifier (unificado)")
        print(f"  Métricas: Binary (precision, recall, f1)")
        print(f"  Validação: Estratificada 5-fold + Holdout 80/20")
        print(f"  Métrica de otimização: PR AUC (Average Precision)")
        print(f"  Tempo estimado: ~30 minutos por modelo")
        print(f"  Resultados salvos em: /Users/i583975/git/tcc/results/")
        print()
        
        # Executa todos os modelos com otimização
        print("🚀 Iniciando experimentos com otimização...")
        results = run_all_models_optimize(X, y, n_trials=N_TRIALS, data_source=data_source, classification_type=classification_type)
    #results = run_single_model("Gradient Boosting", optimize_gradient_boosting_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("Decision Tree", optimize_decision_tree_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("Support Vector Classifier", optimize_svc_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("Multi-Layer Perceptron", optimize_mlp_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("CatBoost", optimize_catboost_classifier, X, y, n_trials=N_TRIALS)

    # Resumo final
    data_source = "renan" if use_renan else "ana"
    classification_type = "multiclass" if use_multiclass else "binary"
    
    if use_default:
        summarize_results(results, mode="default", data_source=data_source, 
                        classification_type=classification_type)
    else:
        summarize_results(results, mode="optimized", data_source=data_source,
                        classification_type=classification_type)
    
    print("\n🎉 EXPERIMENTO CONCLUÍDO!")
    print("💾 Resultados salvos em arquivos organizados por modelo.")


if __name__ == "__main__":
    # Processa argumentos da linha de comando
    args = parse_arguments()
    
    # Executa o experimento com as opções escolhidas
    main(args.renan, args.multiclass, args.default)
