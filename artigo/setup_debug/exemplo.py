#!/usr/bin/env python3
"""
Exemplo de uso das funções de classificação de genes-alvo.
Este script demonstra como usar cada função individualmente.
"""

import sys
sys.path.append('/Users/i583975/git/tcc')

import numpy as np
from processing import prepare_dataset, get_dataset_info
import models
from models import (
    optimize_decision_tree_classifier,
    optimize_random_forest_classifier,
    optimize_gradient_boosting_classifier
)


def example_data_loading():
    """Exemplo de como carregar e preparar os dados"""
    print("🔍 EXEMPLO: Carregamento e Preparação dos Dados")
    print("="*60)
    
    # Caminhos dos arquivos
    features_path = "/Users/i583975/git/tcc/renan/data_files/omics_features/UNION_features.tsv"
    labels_path = "/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv"
    
    # Preparar dataset
    X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path)
    
    if X is not None:
        # Informações do dataset
        info = get_dataset_info(X, y, gene_names, feature_names)
        
        print(f"\n📊 Dataset preparado com sucesso!")
        print(f"   Amostras: {info['n_samples']}")
        print(f"   Features: {info['n_features']}")
        print(f"   Classes: {info['class_distribution']}")
        
        return X, y, gene_names, feature_names
    else:
        print("❌ Erro ao carregar dados")
        return None, None, None, None


def example_single_model(X, y):
    """Exemplo de como executar um modelo individual"""
    print("\n🌳 EXEMPLO: Executando Decision Tree Individual")
    print("="*60)
    
    try:
        # Executa Decision Tree com apenas 5 trials (rápido para exemplo)
        best_model = optimize_decision_tree_classifier(X, y, n_trials=5)
        
        print("✅ Decision Tree executado com sucesso!")
        print(f"   Modelo treinado: {type(best_model).__name__}")
        
        return best_model
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


def example_model_comparison(X, y):
    """Exemplo de comparação entre modelos"""
    print("\n🔬 EXEMPLO: Comparação Rápida entre Modelos")
    print("="*60)
    
    models = [
        ("Decision Tree", optimize_decision_tree_classifier),
        ("Random Forest", optimize_random_forest_classifier),
        ("Gradient Boosting", optimize_gradient_boosting_classifier)
    ]
    
    results = []
    
    for name, optimizer_func in models:
        print(f"\n🚀 Executando {name}...")
        try:
            # Apenas 3 trials para exemplo rápido
            model = optimizer_func(X, y, n_trials=3)
            results.append((name, "✅ Sucesso", model))
        except Exception as e:
            results.append((name, f"❌ Erro: {e}", None))
    
    print("\n📋 RESULTADOS:")
    for name, status, model in results:
        print(f"   {name}: {status}")
    
    return results


def example_custom_evaluation(model, X, y):
    """Exemplo de avaliação customizada"""
    print("\n📊 EXEMPLO: Avaliação Customizada")
    print("="*60)
    
    if model is None:
        print("❌ Modelo não disponível para avaliação")
        return
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Divisão personalizada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=123, stratify=y
        )
        
        # Retreinar modelo (se necessário)
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
        
        # Predições
        y_pred = model.predict(X_test)
        
        print("🎯 Relatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        print("\n🔢 Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))
        
    except Exception as e:
        print(f"❌ Erro na avaliação: {e}")


def main():
    """Executa todos os exemplos"""
    print("🧬 EXEMPLOS DE USO - CLASSIFICAÇÃO DE GENES-ALVO")
    print("="*80)
    
    # 1. Carregar dados
    X, y, gene_names, feature_names = example_data_loading()
    
    if X is None:
        print("❌ Não foi possível carregar os dados. Abortando exemplos.")
        return
    
    # 2. Modelo individual
    best_model = example_single_model(X, y)
    
    # 3. Comparação de modelos
    comparison_results = example_model_comparison(X, y)
    
    # 4. Avaliação customizada
    example_custom_evaluation(best_model, X, y)
    
    print("\n🎉 EXEMPLOS CONCLUÍDOS!")
    print("💡 Para experimentos completos, execute: python main.py")


if __name__ == "__main__":
    main()
