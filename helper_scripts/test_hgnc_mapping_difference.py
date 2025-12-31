#!/usr/bin/env python3
"""
Test script to compare genes loaded with and without HGNC mapping
"""

import pandas as pd
import numpy as np
from src.process_data import load_hgnc_mapping

def load_without_mapping(classification_type='binary'):
    """Load dataset WITHOUT HGNC mapping"""
    features_path = "./data/UNION_features.tsv"
    labels_path = "./data/processed/UNION_labels.tsv"
    
    # Load features and labels
    features_df = pd.read_csv(features_path, sep='\t', index_col=0)
    labels_df = pd.read_csv(labels_path, sep='\t', index_col=0)
    
    # Find common genes
    common_genes = features_df.index.intersection(labels_df.index)
    
    # Filter to common genes
    labels_df = labels_df.loc[common_genes]
    features_df = features_df.loc[common_genes]
    
    # Select label column
    label_column = '3class' if classification_type == 'multiclass' else '2class'
    
    # Filter labeled data
    labeled_mask = labels_df[label_column].notna()
    gene_names = features_df[labeled_mask].index.values
    
    return set(gene_names)


def load_with_mapping(classification_type='binary'):
    """Load dataset WITH HGNC mapping"""
    features_path = "./data/UNION_features.tsv"
    labels_path = "./data/processed/UNION_labels.tsv"
    
    # Load HGNC mapping
    hgnc_mapping, unmatched_genes, withdrawn_genes = load_hgnc_mapping()
    
    # Load features and labels
    features_df = pd.read_csv(features_path, sep='\t', index_col=0)
    labels_df = pd.read_csv(labels_path, sep='\t', index_col=0)
    
    # Apply HGNC mapping to gene names (index) - map where available, keep original otherwise
    features_df.index = features_df.index.map(lambda x: hgnc_mapping.get(x, x))
    labels_df.index = labels_df.index.map(lambda x: hgnc_mapping.get(x, x))
    
    # Only remove genes that are explicitly unmatched or withdrawn
    unmatched_symbols = {g['input'] for g in unmatched_genes}
    withdrawn_symbols = {g['input'] for g in withdrawn_genes}
    genes_to_remove = unmatched_symbols | withdrawn_symbols
    
    features_df = features_df[~features_df.index.isin(genes_to_remove)]
    labels_df = labels_df[~labels_df.index.isin(genes_to_remove)]
    
    # Find common genes
    common_genes = features_df.index.intersection(labels_df.index)
    
    # Filter to common genes
    labels_df = labels_df.loc[common_genes]
    features_df = features_df.loc[common_genes]
    
    # Select label column
    label_column = '3class' if classification_type == 'multiclass' else '2class'
    
    # Filter labeled data - need to filter both DataFrames together
    labeled_mask = labels_df[label_column].notna()
    labels_df_filtered = labels_df[labeled_mask]
    features_df_filtered = features_df.loc[labels_df_filtered.index]
    
    gene_names = features_df_filtered.index.values
    
    return set(gene_names)


def main():
    print("="*80)
    print("COMPARING GENE SETS: WITH vs WITHOUT HGNC MAPPING")
    print("="*80)
    
    for classification_type in ['binary', 'multiclass']:
        print(f"\n{'='*80}")
        print(f"{classification_type.upper()} CLASSIFICATION")
        print(f"{'='*80}")
        
        # Load without mapping
        print("\nLoading WITHOUT HGNC mapping...")
        genes_without = load_without_mapping(classification_type)
        print(f"  Total genes: {len(genes_without)}")
        
        # Load with mapping
        print("\nLoading WITH HGNC mapping...")
        genes_with = load_with_mapping(classification_type)
        print(f"  Total genes: {len(genes_with)}")
        
        # Compare
        print(f"\n{'─'*80}")
        print("COMPARISON:")
        print(f"{'─'*80}")
        
        genes_removed = genes_without - genes_with
        genes_added = genes_with - genes_without
        genes_common = genes_without & genes_with
        
        print(f"  Common genes (in both):     {len(genes_common)}")
        print(f"  Genes removed by mapping:   {len(genes_removed)}")
        print(f"  Genes added by mapping:     {len(genes_added)}")
        print(f"  Net change:                 {len(genes_with) - len(genes_without):+d}")
        
        if genes_removed:
            print(f"\n  Genes REMOVED by HGNC mapping ({len(genes_removed)}):")
            for gene in sorted(list(genes_removed)[:20]):  # Show first 20
                print(f"    - {gene}")
            if len(genes_removed) > 20:
                print(f"    ... and {len(genes_removed) - 20} more")
        
        if genes_added:
            print(f"\n  Genes ADDED by HGNC mapping ({len(genes_added)}):")
            for gene in sorted(list(genes_added)[:20]):  # Show first 20
                print(f"    + {gene}")
            if len(genes_added) > 20:
                print(f"    ... and {len(genes_added) - 20} more")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
