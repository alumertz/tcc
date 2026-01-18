import pandas as pd



'''
    The code below verifies the consistency of gene names across multiple files used in the project.
'''




# Paths to the files
file1_path = '/home/kamille/Documentos/tcc/data/processed/UNION_labels.tsv'
file2_path = '/home/kamille/Documentos/tcc/data/UNION_features.tsv'
file3_path = '/home/kamille/Documentos/tcc/renan/data_files/labels/UNION_labels.tsv'
file4_path = '/home/kamille/Documentos/tcc/data/processed/new_labels_with_features.tsv'

# Load the files
df1 = pd.read_csv(file1_path, sep='\t')
df2 = pd.read_csv(file2_path, sep='\t')
df3 = pd.read_csv(file3_path, sep='\t') 
df4 = pd.read_csv(file4_path, sep='\t')

# Extract unique gene names
genes1 = df1['genes'].dropna().unique()
genes2 = df2['gene'].dropna().unique()
genes3 = df3['gene'].dropna().unique()
genes4 = df4['gene'].dropna().unique()

# Check for duplicates in each file
duplicates1 = df1[df1.duplicated('genes', keep=False)]['genes'].unique() if 'genes' in df1.columns else []
duplicates2 = df2[df2.duplicated('gene', keep=False)]['gene'].unique() if 'gene' in df2.columns else []
duplicates3 = df3[df3.duplicated('gene', keep=False)]['gene'].unique() if 'gene' in df3.columns else []
duplicates4 = df4[df4.duplicated('gene', keep=False)]['gene'].unique() if 'gene' in df4.columns else []

print("Nomes repetidos em UNION_labels.tsv da Ana:")
if len(duplicates1) > 0:
    print(duplicates1)
else:
    print("Nenhum nome repetido.")

print("\nNomes repetidos em UNION_features.tsv do Renan:")
if len(duplicates2) > 0:
    print(duplicates2)
else:
    print("Nenhum nome repetido.")

print("\nNomes repetidos em UNION_labels.tsv do Renan:")
if len(duplicates3) > 0:
    print(duplicates3)
else:
    print("Nenhum nome repetido.")

print("\nNomes repetidos em new_labels_with_features.tsv:")
if len(duplicates4) > 0:
    print(duplicates4)
else:
    print("Nenhum nome repetido.")



# Compare the sets 1 and 2
set1 = set(genes1)
set2 = set(genes2)

genes_only_in_1 = set1 - set2
genes_only_in_2 = set2 - set1

print("\nGenes em UNION_labels.tsv da Ana mas não em UNION_features.tsv do Renan:")
if genes_only_in_1:
    print(sorted(genes_only_in_1))
    print(f"Total: {len(genes_only_in_1)} genes")
else:
    print("Nenhum.")

print("\nGenes em UNION_features.tsv do Renan mas não em UNION_labels.tsv da Ana:")
if genes_only_in_2:
    print(sorted(genes_only_in_2))
    print(f"Total: {len(genes_only_in_2)} genes")
else:
    print("Nenhum.")


# Compare the sets 1 and 3
set3 = set(genes3)

genes_only_in_1_vs_3 = set1 - set3
genes_only_in_3_vs_1 = set3 - set1

print("\nGenes em UNION_labels.tsv da Ana mas não em UNION_labels.tsv do Renan:")

if genes_only_in_1_vs_3:
    print(sorted(genes_only_in_1_vs_3))
    print(f"Total: {len(genes_only_in_1_vs_3)} genes")
else:
    print("Nenhum.")

print("\nGenes em UNION_labels.tsv do Renan mas não em UNION_labels.tsv da Ana:")
if genes_only_in_3_vs_1:
    print(sorted(genes_only_in_3_vs_1))
    print(f"Total: {len(genes_only_in_3_vs_1)} genes")
else:
    print("Nenhum.")


# Compare the sets 2 and 4
set4 = set(genes4)

genes_only_in_2_vs_4 = set2 - set4
genes_only_in_4_vs_2 = set4 - set2

print("\nGenes em UNION_features.tsv do Renan mas não em new_labels_with_features.tsv:")

if genes_only_in_2_vs_4:
    print(sorted(genes_only_in_2_vs_4))
    print(f"Total: {len(genes_only_in_2_vs_4)} genes")
else:
    print("Nenhum.")

print("\nGenes em new_labels_with_features.tsv mas não em UNION_features.tsv do Renan:")
if genes_only_in_4_vs_2:
    print(sorted(genes_only_in_4_vs_2))
    print(f"Total: {len(genes_only_in_4_vs_2)} genes")
else:
    print("Nenhum.")

