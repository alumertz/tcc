import pandas as pd



'''
    The code below performs several data cleaning steps on gene labels and features.
'''




'''
    Removes genes which are inexistent in renan/data_files/labels/UNION_labels.tsv but present in data/processed/UNION_labels.tsv
'''

# Load the files
original_labels_path = '/home/kamille/Documentos/tcc/renan/data_files/labels/UNION_labels.tsv'
processed_labels_path = '/home/kamille/Documentos/tcc/data/processed/UNION_labels.tsv'  

df_original = pd.read_csv(original_labels_path, sep='\t')
df_processed = pd.read_csv(processed_labels_path, sep='\t')

# Get the set of genes in the original labels
original_genes = set(df_original['gene'])

# Filter the processed dataframe to keep only genes present in the original labels
df_filtered = df_processed[df_processed['genes'].isin(original_genes)]

# Print the genes removed
removed_genes = set(df_processed['genes']) - set(df_filtered['genes'])

if removed_genes:
    print(f"Quantidade: {len(removed_genes)}")
    print(f"\nGenes removidos devido a inexistência no arquivo original: \n{removed_genes}")





'''
    Removes genes with empty 2class or 3class values from UNION_labels.tsv based on the filtered labels above
'''

# Further filter the dataframe to remove rows with empty 2class or 3class
df_labels = df_filtered.dropna(subset=['2class', '3class'])

print(f"\n\n\nQuantidade de genes após remoção de valores vazios em 2class ou 3class: {df_labels.shape[0]}")

# Save to new file
output_path = '/home/kamille/Documentos/tcc/data/processed/UNION_labels_filtered.tsv'
df_labels.to_csv(output_path, sep='\t', index=False)

print(f"\nNovo arquivo criado em {output_path}")





'''
    Removes genes with empty 2class or 3class values from UNION_features.tsv based on the filtered labels above
'''

# Load the features file
features_path = '/home/kamille/Documentos/tcc/data/UNION_features.tsv'
df_features = pd.read_csv(features_path, sep='\t')

# Get the set of genes to keep from the filtered labels
genes_to_keep = set(df_labels['genes'])

# Filter the features dataframe to keep only genes present in the filtered labels
df_features_filtered = df_features[df_features['gene'].isin(genes_to_keep)]
print(f"\n\n\nQuantidade de genes em features após remoção de genes não presentes nas labels filtradas: {df_features_filtered.shape[0]}")

# Save to new file
output_features_path = '/home/kamille/Documentos/tcc/data/UNION_features_filtered.tsv'
df_features_filtered.to_csv(output_features_path, sep='\t', index=False)
print(f"\nNovo arquivo de features criado em {output_features_path}")





'''
    hecks if the genes removed from labels and features are the same
'''

# Get the set of genes in the filtered features
filtered_feature_genes = set(df_features_filtered['gene'])

# Find genes that are in labels but not in features
labels_not_in_features = genes_to_keep - filtered_feature_genes

if labels_not_in_features:
    print(f"\n\n\nGenes presentes em labels filtradas mas ausentes em features filtradas: \n{labels_not_in_features}")
else:
    print("\n\n\nTodos os genes nas labels filtradas estão presentes nas features filtradas.")

# Find genes that are in features but not in labels
features_not_in_labels = filtered_feature_genes - genes_to_keep 

if features_not_in_labels:
    print(f"\nGenes presentes em features filtradas mas ausentes em labels filtradas: \n{features_not_in_labels}")
else:
    print("\nTodos os genes nas features filtradas estão presentes nas labels filtradas.")

