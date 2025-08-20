#!/usr/bin/env python3

import sys
sys.path.append('/Users/i583975/git/tcc/artigo')
from processamento import prepare_dataset

# Testa o dataset completo
features_path = "/Users/i583975/git/tcc/renan/data_files/omics_features/UNION_features.tsv"
labels_path = "/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv"

print("=== TESTANDO DATASET COMPLETO ===")
X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path)

if X is not None:
    print(f"\n=== RESUMO FINAL ===")
    print(f"✅ Total de amostras: {X.shape[0]}")
    print(f"✅ Total de features: {X.shape[1]}")
    print(f"✅ Distribuição das classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Verificações adicionais
    import numpy as np
    unique, counts = np.unique(y, return_counts=True)
    print(f"✅ Classes encontradas: {unique}")
    print(f"✅ Contagens por classe: {counts}")
    print(f"✅ Razão positivos/negativos: {counts[1]/counts[0]:.4f}")
    print(f"✅ Percentual da classe minoritária: {counts[1]/sum(counts)*100:.2f}%")
else:
    print("❌ Erro no processamento do dataset!")
