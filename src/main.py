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
from process_data import get_canonical_genes, get_candidate_genes
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
from reports import summarize_optimized_results, summarize_default_results
from evaluation import evaluate_classification_on_test
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
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
        features_path = "../renan/data_files/omics_features/UNION_features.tsv"
        labels_path = "../renan/data_files/labels/UNION_labels.tsv"
        data_source = "RENAN"
        print(f"📁 Usando dados do RENAN:")
        print(f"   Features: renan/data_files/omics_features/UNION_features.tsv")
        print(f"   Labels: renan/data_files/labels/UNION_labels.tsv")
        print(f"   Formato labels: gene, label (True/False/NaN)")
    else:
        features_path = "../data/UNION_features.tsv"
        labels_path = "../data/processed/UNION_labels.tsv"
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
        best_model, test_metrics = optimizer_func(X, y, n_trials=n_trials, save_results=True)
        
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


def evaluate_model_default(model, model_name, X, y):
    """
    Avalia um modelo com parâmetros padrão usando holdout e 5-fold CV
    Pipeline unificado: StandardScaler + Classifier, métricas binary
    
    Args:
        model: Modelo do scikit-learn com parâmetros padrão
        model_name (str): Nome do modelo
        X (np.array): Features
        y (np.array): Labels
        
    Returns:
        dict: Resultados da avaliação
    """
    print(f"\n{'='*80}")
    print(f"AVALIANDO MODELO: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Divisão treino/teste (mesmo random_state do main.py otimizado)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset dividido:")
    print(f"  Treino+Val: {X_trainval.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    
    # Pipeline unificado: SEMPRE com StandardScaler
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])
    
    # Validação cruzada 5-fold no conjunto treino+validação
    print("\nExecutando validação cruzada 5-fold...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
    
    # Métricas para validação cruzada - BINARY (consistente com otimizado)
    cv_scores = {
        'accuracy': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='accuracy'),
        'precision': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='precision_binary'),
        'recall': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='recall_binary'),
        'f1': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='f1_binary'),
        'roc_auc': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='roc_auc'),
        'pr_auc': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='average_precision')
    }
    
    # Calcular médias e desvios padrão
    cv_results = {}
    for metric, scores in cv_scores.items():
        cv_results[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores.tolist()
        }
    
    print("Resultados da validação cruzada:")
    for metric, result in cv_results.items():
        print(f"  {metric.upper()}: {result['mean']:.4f} ± {result['std']:.4f}")
    
    # Treinar no conjunto treino+validação completo e avaliar no teste
    print("\nTreinando no conjunto completo e avaliando no teste...")
    pipeline.fit(X_trainval, y_trainval)
    
    # Avaliação no conjunto de teste usando função unificada
    test_metrics = evaluate_classification_on_test(pipeline, X_test, y_test, return_dict=True)
    
    print("Resultados no conjunto de teste:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Obter predições para o relatório de classificação
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Relatório de classificação detalhado
    from reports import generate_enhanced_classification_report
    class_report = generate_enhanced_classification_report(y_test, y_pred, y_pred_proba)
    print(f"\nRelatório de classificação:\n{class_report}")
    
    # Salvar resultados (usando estrutura similar ao otimizado)
    save_default_results(model_name, cv_results, test_metrics, class_report, pipeline.get_params())
    
    results = {
        'model_name': model_name,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'classification_report': class_report,
        'model': pipeline,
        'status': 'success'
    }
    
    return results


def save_default_results(model_name, cv_results, test_metrics, class_report, params):
    """
    Salva resultados dos modelos padrão em estrutura similar aos otimizados
    """
    import json
    from datetime import datetime
    
    # Criar diretório do modelo
    model_dir = f"results/{model_name.lower().replace(' ', '_')}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Timestamp para arquivo único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Estrutura similar aos modelos otimizados
    results_data = {
        'model_name': model_name,
        'mode': 'default_parameters',
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'parameters': params,
        'timestamp': timestamp
    }
    
    # Salvar métricas em JSON
    metrics_file = f"{model_dir}/default_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Salvar relatório em texto
    report_file = f"{model_dir}/default_results_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(f"MODELO: {model_name} (Parâmetros Padrão)\n")
        f.write("="*80 + "\n\n")
        f.write("VALIDAÇÃO CRUZADA (5-fold):\n")
        for metric, result in cv_results.items():
            f.write(f"  {metric.upper()}: {result['mean']:.4f} ± {result['std']:.4f}\n")
        f.write("\nTESTE FINAL:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric.upper()}: {value:.4f}\n")
        f.write(f"\nRELATÓRIO DE CLASSIFICAÇÃO:\n{class_report}\n")
        f.write(f"\nPARÂMETROS:\n{json.dumps(params, indent=2)}\n")


def run_all_default_models(X, y):
    """
    Executa todos os modelos com parâmetros padrão
    Pipeline unificado para todos: StandardScaler + Classifier
    
    Args:
        X (np.array): Features
        y (np.array): Labels
        
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
        ("Support Vector Classifier", SVC(probability=True, random_state=42))
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
            result = evaluate_model_default(model, model_name, X, y)
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
        results = run_all_default_models(X, y)
        
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
        results = run_all_models(X, y, n_trials=N_TRIALS)
    #results = run_single_model("Gradient Boosting", optimize_gradient_boosting_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("Decision Tree", optimize_decision_tree_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("Support Vector Classifier", optimize_svc_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("Multi-Layer Perceptron", optimize_mlp_classifier, X, y, n_trials=N_TRIALS)
    #results = run_single_model("CatBoost", optimize_catboost_classifier, X, y, n_trials=N_TRIALS)

    # Resumo final
    if use_default:
        summarize_default_results(results)
    else:
        summarize_optimized_results(results)
    
    print("\n🎉 EXPERIMENTO CONCLUÍDO!")
    print("💾 Resultados salvos em arquivos organizados por modelo.")


if __name__ == "__main__":
    # Processa argumentos da linha de comando
    args = parse_arguments()
    
    # Executa o experimento com as opções escolhidas
    main(args.renan, args.multiclass, args.default)
