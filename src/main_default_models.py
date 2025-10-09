#!/usr/bin/env python3
"""
Arquivo para testar modelos com parâmetros padrão
para comparação com os modelos otimizados via Optuna.
"""

import sys
import os
sys.path.append('/Users/i583975/git/tcc')

import numpy as np
import pandas as pd
from src.processing import prepare_dataset, get_dataset_info, split_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


def evaluate_model_default(model, model_name, X, y, save_results=True):
    """
    Avalia um modelo com parâmetros padrão usando holdout e 5-fold CV
    
    Args:
        model: Modelo do scikit-learn
        model_name (str): Nome do modelo
        X (np.array): Features
        y (np.array): Labels
        save_results (bool): Se deve salvar os resultados
        
    Returns:
        dict: Resultados da avaliação
    """
    print(f"\n{'='*80}")
    print(f"AVALIANDO MODELO: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Divisão treino/teste (mesmo random_state do main.py)
    from sklearn.model_selection import train_test_split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset dividido:")
    print(f"  Treino+Val: {X_trainval.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    
    # Configurar pipeline com escalonamento se necessário
    # if model_name.lower() in ['mlp', 'knn']:
    #     pipeline = Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("classifier", model)
    #     ])
    # else:
    pipeline = Pipeline([
        ("classifier", model)
    ])
    
    # Validação cruzada 5-fold no conjunto treino+validação
    print("\nExecutando validação cruzada 5-fold...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
    
    # Métricas para validação cruzada
    cv_scores = {
        'accuracy': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='accuracy'),
        'precision': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='precision'),
        'recall': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='recall'),
        'f1': cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring='f1'),
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
    
    # Predições no conjunto de teste
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Métricas no conjunto de teste
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    print("Resultados no conjunto de teste:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Relatório de classificação detalhado
    class_report = classification_report(y_test, y_pred, target_names=['Non-driver', 'Driver'])
    print(f"\nRelatório de classificação:\n{class_report}")
    
    # Salvar resultados se solicitado
    if save_results:
        save_default_results(model_name, cv_results, test_metrics, class_report, pipeline.get_params())
    
    results = {
        'model_name': model_name,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'classification_report': class_report,
        'model': pipeline
    }
    
    return results


def save_default_results(model_name, cv_results, test_metrics, class_report, params):
    """
    Salva os resultados em arquivos organizados por modelo
    """
    # Criar diretório base
    base_dir = "../results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Criar diretório específico do modelo
    model_dir = os.path.join(base_dir, model_name.lower().replace(' ', '_').replace('-', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    # Timestamp para arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar resultados em arquivo de texto
    results_file = os.path.join(model_dir, f"default_results_{timestamp}.txt")
    
    with open(results_file, 'w') as f:
        f.write(f"RESULTADOS DO MODELO: {model_name.upper()} (PARÂMETROS PADRÃO)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Parâmetros utilizados
        f.write("PARÂMETROS UTILIZADOS (PADRÃO):\n")
        f.write("-"*50 + "\n")
        classifier_params = {k.replace('classifier__', ''): v for k, v in params.items() if k.startswith('classifier__')}
        for param, value in classifier_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        # Resultados da validação cruzada
        f.write("RESULTADOS DA VALIDAÇÃO CRUZADA (5-FOLD):\n")
        f.write("-"*50 + "\n")
        for metric, result in cv_results.items():
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Média: {result['mean']:.4f}\n")
            f.write(f"  Desvio Padrão: {result['std']:.4f}\n")
            f.write(f"  Scores por fold: {result['scores']}\n\n")
        
        # Métricas no conjunto de teste
        f.write("AVALIAÇÃO NO CONJUNTO DE TESTE FINAL:\n")
        f.write("-"*50 + "\n")
        f.write(f"Acurácia: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precisão: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {test_metrics['f1_score']:.4f}\n")
        f.write(f"ROC AUC: {test_metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC: {test_metrics['pr_auc']:.4f}\n\n")
        
        # Relatório de classificação
        f.write("RELATÓRIO DETALHADO:\n")
        f.write("-"*30 + "\n")
        f.write(class_report)
        f.write("\n")
    
    # Salvar resultados em JSON para análises posteriores
    json_file = os.path.join(model_dir, f"default_metrics_{timestamp}.json")
    
    json_data = {
        'model_name': model_name,
        'timestamp': timestamp,
        'parameters': classifier_params,
        'cv_results': cv_results,
        'test_metrics': test_metrics
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Resultados salvos em:")
    print(f"  Arquivo texto: {results_file}")
    print(f"  Arquivo JSON: {json_file}")


def run_all_default_models(X, y):
    """
    Executa todos os modelos com parâmetros padrão
    
    Args:
        X (np.array): Features
        y (np.array): Labels
        
    Returns:
        list: Lista com resultados de todos os modelos
    """
    # Configuração dos modelos com parâmetros padrão
    models_config = [
        ("Decision Tree", DecisionTreeClassifier(random_state=30)),
        ("Random Forest", RandomForestClassifier(random_state=30)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=30)),
        ("Histogram Gradient Boosting", HistGradientBoostingClassifier(random_state=30)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Multi-Layer Perceptron", MLPClassifier(random_state=30, max_iter=1000)),
        # SVC não incluído conforme solicitado
    ]
    
    results = []
    
    print("INICIANDO AVALIAÇÃO COM PARÂMETROS PADRÃO")
    print(f"Dataset: {X.shape[0]} amostras x {X.shape[1]} features")
    print(f"Modelos a avaliar: {len(models_config)}")
    print()
    
    for i, (model_name, model) in enumerate(models_config, 1):
        print(f"Progresso: {i}/{len(models_config)} modelos")
        
        try:
            result = evaluate_model_default(model, model_name, X, y, save_results=True)
            result['status'] = 'success'
            results.append(result)
            print(f"✅ {model_name} executado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao executar {model_name}: {e}")
            results.append({
                'model_name': model_name,
                'status': 'error',
                'error': str(e)
            })
        
        # Breve pausa entre modelos
        import time
        time.sleep(1)
    
    return results


def summarize_default_results(results):
    """
    Cria um resumo dos resultados dos modelos padrão
    """
    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS (PARÂMETROS PADRÃO)")
    print("="*80)
    
    successful_models = [r for r in results if r['status'] == 'success']
    failed_models = [r for r in results if r['status'] == 'error']
    
    print(f"✅ Modelos executados com sucesso: {len(successful_models)}")
    print(f"❌ Modelos com erro: {len(failed_models)}")
    print()
    
    if successful_models:
        print("COMPARAÇÃO DE PERFORMANCE (CONJUNTO DE TESTE):")
        print("-" * 80)
        print(f"{'Modelo':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'ROC AUC':<9} {'PR AUC':<8}")
        print("-" * 80)
        
        for result in successful_models:
            metrics = result['test_metrics']
            print(f"{result['model_name']:<25} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<11.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['f1_score']:<8.4f} "
                  f"{metrics['roc_auc']:<9.4f} "
                  f"{metrics['pr_auc']:<8.4f}")
    
    if failed_models:
        print("\nMODELOS COM ERRO:")
        for result in failed_models:
            print(f"  • {result['model_name']}: {result['error']}")
    
    print("="*80)


def main():
    """
    Função principal do experimento com parâmetros padrão
    """
    print("AVALIAÇÃO DE GENES-ALVO COM PARÂMETROS PADRÃO")
    print("="*80)
    
    # Caminhos para os arquivos de dados (mesmos do main.py)
    features_path = "../data/UNION_features.tsv"
    labels_path = "../data/processed/UNION_labels.tsv"
    
    # Verifica se os arquivos existem
    if not os.path.exists(features_path):
        print(f"Arquivo de features não encontrado: {features_path}")
        return
    
    if not os.path.exists(labels_path):
        print(f"Arquivo de labels não encontrado: {labels_path}")
        return
    
    # Prepara o dataset (mesma função do main.py)
    print("Carregando e preparando dados...")
    X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path)
    
    if X is None:
        print("Erro ao preparar dataset. Abortando.")
        return
    
    # Mostra informações do dataset
    dataset_info = get_dataset_info(X, y, gene_names, feature_names)
    print("\nINFORMAÇÕES DO DATASET:")
    print(f"  Amostras: {dataset_info['n_samples']}")
    print(f"  Features: {dataset_info['n_features']}")
    print(f"  Distribuição das classes: {dataset_info['class_distribution']}")
    print(f"  Estatísticas das features:")
    print(f"    - Média: {dataset_info['feature_stats']['mean']:.4f}")
    print(f"    - Desvio padrão: {dataset_info['feature_stats']['std']:.4f}")
    print(f"    - Valores zero: {dataset_info['feature_stats']['zeros_percentage']:.2f}%")
    
    print(f"\nCONFIGURAÇÃO DO EXPERIMENTO:")
    print(f"  Parâmetros: PADRÃO (sem otimização)")
    print(f"  Validação: Estratificada 5-fold + Holdout 80/20")
    print(f"  Métricas: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC")
    print(f"  Resultados salvos em: /Users/i583975/git/tcc/artigo/results/ (com 'default' no nome)")
    print()
    
    # Executa todos os modelos com parâmetros padrão
    print("Iniciando experimentos com parâmetros padrão...")
    results = run_all_default_models(X, y)
    
    # Resumo final
    summarize_default_results(results)
    
    print("\nEXPERIMENTO COM PARÂMETROS PADRÃO CONCLUÍDO!")
    print("Resultados salvos em arquivos organizados por modelo com 'default' no nome.")


if __name__ == "__main__":
    main()
