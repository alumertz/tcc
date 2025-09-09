#!/usr/bin/env python3
"""
Teste rápido de um modelo com o novo sistema de salvamento
"""

import sys
sys.path.append('/Users/i583975/git/tcc')

import numpy as np
from processing import prepare_dataset
from models import optimize_decision_tree_classifier

def test_single_model():
    """Testa um único modelo com poucos trials"""
    print("🧪 TESTE RÁPIDO - DECISION TREE")
    print("="*50)
    
    # Caminhos para os dados
    features_path = "/Users/i583975/git/tcc/renan/data_files/omics_features/UNION_features.tsv"
    labels_path = "/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv"
    
    # Carrega dados
    print("📂 Carregando dados...")
    X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path)
    
    if X is None:
        print("❌ Erro ao carregar dados!")
        return
    
    print(f"✅ Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Testa o modelo com poucos trials
    print("\n🔄 Executando Decision Tree com 3 trials...")
    try:
        model = optimize_decision_tree_classifier(X, y, n_trials=3, save_results=True)
        print("✅ Teste concluído com sucesso!")
        print("💾 Verifique os resultados em: /Users/i583975/git/tcc/artigo/results/decision_tree/")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")

if __name__ == "__main__":
    test_single_model()
