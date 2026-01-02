import pandas as pd

# Load the files
labels_path = '/home/kamille/Documentos/tcc/data/processed/UNION_labels.tsv'
features_path = '/home/kamille/Documentos/tcc/renan/data_files/omics_features/UNION_features.tsv'

df_labels = pd.read_csv(labels_path, sep='\t')
df_features = pd.read_csv(features_path, sep='\t')

# Set gene as index for features for quick lookup
df_features.set_index('gene', inplace=True)

# Function to map 3class to classe
def map_classe(value):
    if pd.isna(value):
        return 'Candidate'
    elif value == 0.0:
        return 'Passenger'
    elif value == 1.0:
        return 'TSG'
    elif value == 2.0:
        return 'ONC'
    else:
        return 'Unknown'  # In case of unexpected values

# Prepare the new dataframe
new_data = []

for _, row in df_labels.iterrows():
    gene = row['genes']
    classe = map_classe(row['3class'])
    
    if gene in df_features.index:
        atributos = ','.join(map(str, df_features.loc[gene].values))
    else:
        atributos = 'NaN'
    new_data.append({'gene': gene, 'classe': classe, 'atributos': atributos})

df_new = pd.DataFrame(new_data)

# Save to new file
output_path = '/home/kamille/Documentos/tcc/data/processed/new_labels_with_features.tsv'
df_new.to_csv(output_path, sep='\t', index=False)

print(f"Novo arquivo criado em {output_path}")