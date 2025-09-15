import pandas as pd
import numpy as np

ONCOKB_FILE = './data/onkoKB.tsv'
NCG_FILE = './data/NCG_cancerdrivers_annotation_supporting_evidence.tsv'
COSMIC_FILE = './data/cosmic.tsv'
OUTPUT_FILE = 'processed_genes_list.csv'

def process_oncokb(type, file_path = ONCOKB_FILE):
    """
    Carrega e processa a lista de genes do OncoKB.
    Filtra apenas os genes que são oncogenes ou TSGs.
    """
    print("Processando OncoKB...")
    try:
        # Carrega o arquivo, que é um TSV (Tab-Separated Values)
        df = pd.read_csv(file_path, sep='\t')

        # Seleciona e renomeia as colunas de interesse
        df = df[['Hugo Symbol', 'Is Oncogene', 'Is Tumor Suppressor Gene']].rename(columns={
            'Hugo Symbol': 'symbol',
            'Is Oncogene': 'Oncogene',
            'Is Tumor Suppressor Gene': 'TSG'
        })

        if type == 'canonical':
            # Filtra para manter apenas os genes que são Oncogene 'Yes' OU TSG 'Yes'
            df_filtered = df[(df['Oncogene'] == 'Yes') | (df['TSG'] == 'Yes')].copy()
        elif type == 'candidate':
            # Filtra para manter apenas os genes que são Oncogene 'No' e TSG 'No'
            df_filtered = df[(df['Oncogene'] == 'No') & (df['TSG'] == 'No')].copy()


        print(f"OncoKB: Encontrados {len(df_filtered)} genes {type}.")
        return df_filtered

    except FileNotFoundError:
        print(f"Erro: O arquivo OncoKB '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])

def process_ncg_canonical(type, file_path = NCG_FILE):
    """
    Carrega e processa a lista de genes do Network of Cancer Genes (NCG).
    Usa as colunas 'NCG_oncogene' e 'NCG_tsg' para determinar o papel do gene.
    """
    print("Processando NCG...")
    try:
        df = pd.read_csv(file_path, sep='\t')

        if type == 'canonical':
            # Filtra para manter genes que são marcados como oncogene (1.0) ou tsg (1.0)
            df_filtered = df[(df['NCG_oncogene'] == 1.0) | (df['NCG_tsg'] == 1.0)].copy()
        elif type == 'candidate':
            # Filtra para manter apenas os genes que são Oncogene 'No' e TSG 'No'
            df_filtered = df[(df['NCG_oncogene'] == 0) & (df['NCG_tsg'] == 0)].copy()

        # Converte as colunas de 0/1 para 'No'/'Yes'
        df_filtered['Oncogene'] = np.where(df_filtered['NCG_oncogene'] == 1.0, 'Yes', 'No')
        df_filtered['TSG'] = np.where(df_filtered['NCG_tsg'] == 1.0, 'Yes', 'No')

        # Seleciona as colunas finais
        df_final = df_filtered[['symbol', 'Oncogene', 'TSG']]

        # Remove duplicatas, mantendo a primeira ocorrência do símbolo do gene
        df_final = df_final.drop_duplicates(subset=['symbol']).reset_index(drop=True)

        print(f"NCG: Encontrados {len(df_final)} {type}.")
        return df_final

    except FileNotFoundError:
        print(f"Erro: O arquivo NCG '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])

def process_cgc(tier: int, file_path = COSMIC_FILE):
    """
    Carrega e processa a lista do COSMIC Cancer Gene Census (CGC).
    Filtra por Tier 1 e interpreta a coluna 'Role in Cancer'.
    """
    print("Processando COSMIC Cancer Gene Census...")
    try:
        df = pd.read_csv(file_path, sep='\t')

        # Filtra para manter apenas os genes de 'Tier 1'
        df_tier1 = df[df['Tier'] == tier].copy()

        # Converte a coluna 'Role in Cancer' para string para evitar erros e lida com valores nulos
        df_tier1['Role in Cancer'] = df_tier1['Role in Cancer'].astype(str).str.lower()

        # Determina o papel do gene baseado na descrição em 'Role in Cancer'
        df_tier1['Oncogene'] = np.where(df_tier1['Role in Cancer'].str.contains('oncogene'), 'Yes', 'No')
        df_tier1['TSG'] = np.where(df_tier1['Role in Cancer'].str.contains('tsg'), 'Yes', 'No')

        # Seleciona e renomeia as colunas
        df_final = df_tier1[['Gene Symbol', 'Oncogene', 'TSG']].rename(columns={'Gene Symbol': 'symbol'})

        # Remove duplicatas
        df_final = df_final.drop_duplicates(subset=['symbol']).reset_index(drop=True)

        print(f"CGC (Tier {tier}): Encontrados {len(df_final)} genes.")
        return df_final
        
    except FileNotFoundError:
        print(f"Erro: O arquivo CGC '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])
    
def process_canonical():
    # Processa cada arquivo de dados
    oncokb_genes = process_oncokb(type = "canonical", file = ONCOKB_FILE)
    ncg_genes = process_ncg_canonical(NCG_FILE)
    cgc_genes = process_cgc(tier = 1, file_path = COSMIC_FILE)

    # Combina as três listas de genes em um único DataFrame
    print("\nCombinando as listas de genes...")
    combined_df = pd.concat([oncokb_genes, ncg_genes, cgc_genes], ignore_index=True)
    print(f"Total de entradas antes da deduplicação: {len(combined_df)}")

    # Remove linhas onde o símbolo do gene é nulo, caso existam
    combined_df.dropna(subset=['symbol'], inplace=True)
    
    # Agrupa por símbolo de gene e consolida as informações de 'Oncogene' e 'TSG'.
    # Se um gene for 'Yes' em qualquer fonte, ele será 'Yes' na lista final.
    final_genes = combined_df.groupby('symbol').agg({
        'Oncogene': lambda x: 'Yes' if 'Yes' in x.values else 'No',
        'TSG': lambda x: 'Yes' if 'Yes' in x.values else 'No'
    }).reset_index()

    # Ordena a lista final por símbolo de gene
    final_genes_sorted = final_genes.sort_values(by='symbol').reset_index(drop=True)

    # Salva o resultado em um arquivo CSV
    try:
        final_genes_sorted.to_csv(OUTPUT_FILE, index=False)
        print(f"\nProcesso concluído com sucesso!")
        print(f"A lista final com {len(final_genes_sorted)} genes canônicos foi salva em '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"\nOcorreu um erro ao salvar o arquivo final: {e}")

    # Exibe as primeiras linhas do DataFrame final
    print("\nAmostra da lista final de genes canônicos:")
    print(final_genes_sorted.head())


def process_candidates():
    """
    Função para processar candidatos
    """
    print("Processando candidatos")
    cgc_genes = process_cgc(tier = 2, file_path = COSMIC_FILE)
    oncokb_genes = process_oncokb(type = "candidate", file_path = ONCOKB_FILE)
    ncg_genes = process_ncg_canonical(type = "candidate", file_path = NCG_FILE)

    pass