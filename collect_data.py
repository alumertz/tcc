import pandas as pd
import numpy as np

def process_oncokb(file_path):
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

        # Filtra para manter apenas os genes que são Oncogene 'Yes' OU TSG 'Yes'
        df_filtered = df[(df['Oncogene'] == 'Yes') | (df['TSG'] == 'Yes')].copy()

        print(f"OncoKB: Encontrados {len(df_filtered)} genes canônicos.")
        return df_filtered

    except FileNotFoundError:
        print(f"Erro: O arquivo OncoKB '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])

def process_ncg(file_path):
    """
    Carrega e processa a lista de genes do Network of Cancer Genes (NCG).
    Usa as colunas 'NCG_oncogene' e 'NCG_tsg' para determinar o papel do gene.
    """
    print("Processando NCG...")
    try:
        df = pd.read_csv(file_path, sep='\t')

        # Filtra para manter genes que são marcados como oncogene (1.0) ou tsg (1.0)
        df_filtered = df[(df['NCG_oncogene'] == 1.0) | (df['NCG_tsg'] == 1.0)].copy()

        # Converte as colunas de 0/1 para 'No'/'Yes'
        df_filtered['Oncogene'] = np.where(df_filtered['NCG_oncogene'] == 1.0, 'Yes', 'No')
        df_filtered['TSG'] = np.where(df_filtered['NCG_tsg'] == 1.0, 'Yes', 'No')

        # Seleciona as colunas finais
        df_final = df_filtered[['symbol', 'Oncogene', 'TSG']]

        # Remove duplicatas, mantendo a primeira ocorrência do símbolo do gene
        df_final = df_final.drop_duplicates(subset=['symbol']).reset_index(drop=True)

        print(f"NCG: Encontrados {len(df_final)} genes canônicos.")
        return df_final

    except FileNotFoundError:
        print(f"Erro: O arquivo NCG '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])


def process_cgc(file_path):
    """
    Carrega e processa a lista do COSMIC Cancer Gene Census (CGC).
    Filtra por Tier 1 e interpreta a coluna 'Role in Cancer'.
    """
    print("Processando COSMIC Cancer Gene Census...")
    try:
        df = pd.read_csv(file_path, sep='\t')

        # Filtra para manter apenas os genes de 'Tier 1'
        df_tier1 = df[df['Tier'] == 1].copy()

        # Converte a coluna 'Role in Cancer' para string para evitar erros e lida com valores nulos
        df_tier1['Role in Cancer'] = df_tier1['Role in Cancer'].astype(str).str.lower()

        # Determina o papel do gene baseado na descrição em 'Role in Cancer'
        df_tier1['Oncogene'] = np.where(df_tier1['Role in Cancer'].str.contains('oncogene'), 'Yes', 'No')
        df_tier1['TSG'] = np.where(df_tier1['Role in Cancer'].str.contains('tsg'), 'Yes', 'No')

        # Seleciona e renomeia as colunas
        df_final = df_tier1[['Gene Symbol', 'Oncogene', 'TSG']].rename(columns={'Gene Symbol': 'symbol'})

        # Remove duplicatas
        df_final = df_final.drop_duplicates(subset=['symbol']).reset_index(drop=True)

        print(f"CGC (Tier 1): Encontrados {len(df_final)} genes canônicos.")
        return df_final
        
    except FileNotFoundError:
        print(f"Erro: O arquivo CGC '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])