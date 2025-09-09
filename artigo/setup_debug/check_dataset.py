#!/usr/bin/env python3
import sys
sys.path.append('/Users/i583975/git/tcc')
from processing import prepare_dataset
import numpy as np

features_path = '/Users/i583975/git/tcc/renan/data_files/omics_features/UNION_features.tsv'
labels_path = '/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv'

print('🔍 ANÁLISE DO DATASET')
print('='*50)

X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path)

if X is not None:
    print(f'📊 Dataset shape: {X.shape}')
    unique, counts = np.unique(y, return_counts=True)
    print(f'📈 Classes: {dict(zip(unique, counts))}')
    
    minority_class_size = min(counts)
    print(f'⚠️  Menor classe tem {minority_class_size} amostras')
    print(f'⚠️  Com 5-fold CV, cada fold teria ~{minority_class_size/5:.1f} amostras da classe minoritária')
    
    if minority_class_size < 5:
        print('❌ PROBLEMA: Classe minoritária tem muito poucas amostras para 5-fold CV!')
    elif minority_class_size < 10:
        print('⚠️  AVISO: Classe minoritária pode causar problemas no CV. Recomenda-se 3-fold.')
    else:
        print('✅ Dataset adequado para 5-fold CV')
else:
    print('❌ Erro ao carregar dados')
