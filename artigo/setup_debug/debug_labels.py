#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/i583975/git/tcc/artigo')
from processamento import load_union_labels

# Testa o carregamento dos labels
labels_path = "/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv"

print("=== CARREGANDO LABELS ===")
labels_df = load_union_labels(labels_path)

if labels_df is not None:
    print("\n=== ANÁLISE DETALHADA ===")
    print(f"Total de genes após limpeza: {len(labels_df)}")
    print(f"Distribuição final: {labels_df['label'].value_counts()}")
    print(f"Tipos de dados: {labels_df['label'].dtype}")
    print(f"Valores únicos: {labels_df['label'].unique()}")
else:
    print("Erro no carregamento!")
