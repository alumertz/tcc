import pandas as pd
import numpy as np
import os

ONCOKB_FILE = './data/onkoKB.tsv'
NCG_FILE = './data/NCG_cancerdrivers_annotation_supporting_evidence.tsv'
COSMIC_FILE = './data/cosmic.tsv'
OMIM_FILE = './data/All_Diseases_OMIM.txt'
OUTPUT_DIR = './data/processed'

def process_oncokb(gene_type, file_path=ONCOKB_FILE):
    """
    Carrega e processa a lista de genes do OncoKB.
    Filtra genes baseado no tipo (canonical ou candidate).
    """
    print("Processando OncoKB...")
    try:
        df = pd.read_csv(file_path, sep='\t')
        df = df[['Hugo Symbol', 'Is Oncogene', 'Is Tumor Suppressor Gene']].rename(columns={
            'Hugo Symbol': 'symbol',
            'Is Oncogene': 'Oncogene',
            'Is Tumor Suppressor Gene': 'TSG'
        })

        if gene_type == 'canonical':
            df_filtered = df[(df['Oncogene'] == 'Yes') | (df['TSG'] == 'Yes')].copy()
        elif gene_type == 'candidate':
            df_filtered = df[(df['Oncogene'] == 'No') & (df['TSG'] == 'No')].copy()
        else:
            df_filtered = df.copy()

        print(f"OncoKB: Encontrados {len(df_filtered)} genes {gene_type}.")
        return df_filtered

    except FileNotFoundError:
        print(f"Erro: O arquivo OncoKB '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])

def process_ncg(gene_type, file_path=NCG_FILE):
    """
    Carrega e processa a lista de genes do Network of Cancer Genes (NCG).
    """
    print("Processando NCG...")
    try:
        df = pd.read_csv(file_path, sep='\t')

        if gene_type == 'canonical':
            df_filtered = df[(df['NCG_oncogene'] == 1.0) | (df['NCG_tsg'] == 1.0)].copy()
        elif gene_type == 'candidate':
            df_filtered = df[(df['NCG_oncogene'] == 0) & (df['NCG_tsg'] == 0)].copy()
        else:
            df_filtered = df.copy()

        df_filtered['Oncogene'] = np.where(df_filtered['NCG_oncogene'] == 1.0, 'Yes', 'No')
        df_filtered['TSG'] = np.where(df_filtered['NCG_tsg'] == 1.0, 'Yes', 'No')
        df_final = df_filtered[['symbol', 'Oncogene', 'TSG']]
        df_final = df_final.drop_duplicates(subset=['symbol']).reset_index(drop=True)

        print(f"NCG: Encontrados {len(df_final)} genes {gene_type}.")
        return df_final

    except FileNotFoundError:
        print(f"Erro: O arquivo NCG '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])

def process_cgc(tier, file_path=COSMIC_FILE):
    """
    Carrega e processa a lista do COSMIC Cancer Gene Census (CGC).
    """
    print(f"Processando COSMIC Cancer Gene Census (Tier {tier})...")
    try:
        df = pd.read_csv(file_path, sep='\t')
        df_tier = df[df['Tier'] == tier].copy()
        df_tier['Role in Cancer'] = df_tier['Role in Cancer'].astype(str).str.lower()
        df_tier['Oncogene'] = np.where(df_tier['Role in Cancer'].str.contains('oncogene'), 'Yes', 'No')
        df_tier['TSG'] = np.where(df_tier['Role in Cancer'].str.contains('tsg'), 'Yes', 'No')
        df_final = df_tier[['Gene Symbol', 'Oncogene', 'TSG']].rename(columns={'Gene Symbol': 'symbol'})
        df_final = df_final.drop_duplicates(subset=['symbol']).reset_index(drop=True)

        print(f"CGC (Tier {tier}): Encontrados {len(df_final)} genes.")
        return df_final
        
    except FileNotFoundError:
        print(f"Erro: O arquivo CGC '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])

def process_omim(file_path=OMIM_FILE):
    """
    Carrega e processa a lista de genes do OMIM.
    """
    print("Processando OMIM...")
    try:
        df = pd.read_csv(file_path, sep=' ')
        # Remove aspas das colunas se existirem
        df.columns = df.columns.str.replace('"', '')
        df['SYMBOL'] = df['SYMBOL'].str.replace('"', '')
        
        # Cria DataFrame com formato padrão
        df_omim = pd.DataFrame({
            'symbol': df['SYMBOL'],
            'Oncogene': 'No',  # OMIM não especifica oncogenes/TSGs
            'TSG': 'No'
        })
        
        df_omim = df_omim.drop_duplicates(subset=['symbol']).reset_index(drop=True)
        print(f"OMIM: Encontrados {len(df_omim)} genes.")
        return df_omim

    except FileNotFoundError:
        print(f"Erro: O arquivo OMIM '{file_path}' não foi encontrado.")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG'])

def get_canonical_genes():
    """
    Processa dados para obter genes canônicos (oncogenes e TSGs conhecidos).
    """
    print("\n=== PROCESSANDO GENES CANÔNICOS ===")
    
    # Processa cada fonte de dados
    oncokb_genes = process_oncokb('canonical')
    ncg_genes = process_ncg('canonical')
    cgc_genes = process_cgc(tier=1)

    # Combina as listas
    print("\nCombinando as listas de genes canônicos...")
    combined_df = pd.concat([oncokb_genes, ncg_genes, cgc_genes], ignore_index=True)
    print(f"Total de entradas antes da deduplicação: {len(combined_df)}")

    # Remove linhas com símbolos nulos
    combined_df.dropna(subset=['symbol'], inplace=True)
    
    # Agrupa por símbolo e consolida informações
    final_genes = combined_df.groupby('symbol').agg({
        'Oncogene': lambda x: 'Yes' if 'Yes' in x.values else 'No',
        'TSG': lambda x: 'Yes' if 'Yes' in x.values else 'No'
    }).reset_index()

    # FILTRO CRUCIAL: Remove genes que não são nem oncogenes nem TSGs
    final_genes = final_genes[(final_genes['Oncogene'] == 'Yes') | (final_genes['TSG'] == 'Yes')].copy()

    final_genes_sorted = final_genes.sort_values(by='symbol').reset_index(drop=True)
    print(f"Total de genes canônicos únicos (após filtrar No/No): {len(final_genes_sorted)}")
    
    return final_genes_sorted

def get_candidate_genes():
    """
    Processa dados para obter genes candidatos.
    """
    print("\n=== PROCESSANDO GENES CANDIDATOS ===")
    
    # Processa cada fonte de dados
    cgc_genes = process_cgc(tier=2)
    oncokb_genes = process_oncokb('candidate')
    ncg_genes = process_ncg('candidate')
    omim_genes = process_omim()

    # Combina as listas
    print("\nCombinando as listas de genes candidatos...")
    combined_df = pd.concat([cgc_genes, oncokb_genes, ncg_genes, omim_genes], ignore_index=True)
    print(f"Total de entradas antes da deduplicação: {len(combined_df)}")

    # Remove linhas com símbolos nulos
    combined_df.dropna(subset=['symbol'], inplace=True)
    
    # Agrupa por símbolo e consolida informações
    final_genes = combined_df.groupby('symbol').agg({
        'Oncogene': lambda x: 'Yes' if 'Yes' in x.values else 'No',
        'TSG': lambda x: 'Yes' if 'Yes' in x.values else 'No'
    }).reset_index()

    final_genes_sorted = final_genes.sort_values(by='symbol').reset_index(drop=True)
    print(f"Total de genes candidatos únicos: {len(final_genes_sorted)}")
    
    return final_genes_sorted

def create_union_labels(canonical_genes, candidate_genes):
    """
    Cria arquivo UNION_labels.tsv com 3 colunas:
    - genes: símbolo do gene
    - 2class: classe binária (cancer=1, candidato=NaN, passenger=0)
    - 3class: 3classe (TSG=1, OC=2, candidato=NaN, passenger=0)
    """
    print("\n=== CRIANDO UNION_LABELS ===")
    
    # Combina todos os genes mantendo informações de Oncogene/TSG
    all_genes = pd.concat([canonical_genes, candidate_genes], ignore_index=True)
    all_genes = all_genes.drop_duplicates(subset=['symbol']).sort_values(by='symbol').reset_index(drop=True)
    
    # Identifica categorias dos genes
    canonical_symbols = set(canonical_genes['symbol'])
    candidate_symbols = set(candidate_genes['symbol'])
    
    # Listas para armazenar as classificações
    genes = []
    binary_class = []
    multiclass_labels = []
    
    for _, row in all_genes.iterrows():
        symbol = row['symbol']
        oncogene = row['Oncogene']
        tsg = row['TSG']
        
        genes.append(symbol)
        
        if symbol in canonical_symbols:
            # Gene canônico (cancer) = 1 na classificação binária
            binary_class.append(1)
            
            # Classificação 3classe baseada no tipo
            if tsg == 'Yes' and oncogene == 'Yes':
                # Se é tanto TSG quanto oncogene, prioriza TSG
                multiclass_labels.append(1)  # TSG = 1
            elif tsg == 'Yes':
                multiclass_labels.append(1)  # TSG = 1
            elif oncogene == 'Yes':
                multiclass_labels.append(2)  # Oncogene = 2
            else:
                # Gene canônico mas sem classificação clara (não deveria acontecer)
                multiclass_labels.append(1)  # Default para TSG
                
        elif symbol in candidate_symbols:
            # Gene candidato = NaN em ambas classificações
            binary_class.append(np.nan)
            multiclass_labels.append(np.nan)
        else:
            # Gene passenger = 0 em ambas classificações
            binary_class.append(0)
            multiclass_labels.append(0)
    
    # Cria o DataFrame final com as 3 colunas especificadas
    union_labels = pd.DataFrame({
        'genes': genes,
        '2class': binary_class,
        '3class': multiclass_labels
    })
    
    # Estatísticas para relatório
    n_cancer = sum(1 for x in binary_class if x == 1)
    n_passenger = sum(1 for x in binary_class if x == 0)
    n_candidate = sum(1 for x in binary_class if pd.isna(x))
    
    n_tsg = sum(1 for x in multiclass_labels if x == 1)
    n_oncogene = sum(1 for x in multiclass_labels if x == 2)
    n_candidate_multi = sum(1 for x in multiclass_labels if pd.isna(x))
    n_passenger_multi = sum(1 for x in multiclass_labels if x == 0)
    
    print(f"Total de genes no UNION_labels: {len(union_labels)}")
    print(f"\nClassificação Binária:")
    print(f"  Cancer (1): {n_cancer}")
    print(f"  Candidatos (NaN): {n_candidate}")
    print(f"  Passengers (0): {n_passenger}")
    print(f"\nClassificação 3classe:")
    print(f"  TSG (1): {n_tsg}")
    print(f"  Oncogenes (2): {n_oncogene}")
    print(f"  Candidatos (NaN): {n_candidate_multi}")
    print(f"  Passengers (0): {n_passenger_multi}")
    
    return union_labels

def save_results(canonical_genes, candidate_genes, union_labels):
    """
    Salva os resultados nos arquivos de saída.
    """
    print(f"\n=== SALVANDO RESULTADOS EM {OUTPUT_DIR} ===")
    
    # Cria diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Salva genes canônicos
        canonical_path = os.path.join(OUTPUT_DIR, 'canonical_genes.tsv')
        canonical_genes.to_csv(canonical_path, sep='\t', index=False)
        print(f"Genes canônicos salvos em: {canonical_path}")
        
        # Salva genes candidatos
        candidate_path = os.path.join(OUTPUT_DIR, 'candidate_genes.tsv')
        candidate_genes.to_csv(candidate_path, sep='\t', index=False)
        print(f"Genes candidatos salvos em: {candidate_path}")
        
        # Salva UNION_labels com formatação adequada
        union_path = os.path.join(OUTPUT_DIR, 'UNION_labels.tsv')
        union_labels.to_csv(union_path, sep='\t', index=False)
        print(f"UNION_labels salvos em: {union_path}")
        
        print("\nProcesso concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro ao salvar arquivos: {e}")

def main():
    """
    Função principal que executa todo o pipeline de processamento.
    """
    print("INICIANDO PROCESSAMENTO DE DADOS DE GENES")
    print("=" * 50)
    
    # Processa genes canônicos e candidatos
    canonical_genes = get_canonical_genes()
    candidate_genes = get_candidate_genes()
    
    # Cria arquivo UNION_labels
    union_labels = create_union_labels(canonical_genes, candidate_genes)
    
    # Salva todos os resultados
    save_results(canonical_genes, candidate_genes, union_labels)
    
    # Exibe amostras dos resultados
    print("\n=== AMOSTRAS DOS RESULTADOS ===")
    print("\nGenes canônicos (primeiras 10 linhas):")
    print(canonical_genes.head(10))
    print("\nGenes candidatos (primeiras 10 linhas):")
    print(candidate_genes.head(10))
    print("\nUNION_labels (primeiras 15 linhas):")
    print(union_labels.head(15))
    print("\nResumo das classificações:")
    print(f"Classificação binária - Distribuição:")
    print(union_labels['2class'].value_counts(dropna=False))
    print(f"\nClassificação multiclasse - Distribuição:")
    print(union_labels['3class'].value_counts(dropna=False))

if __name__ == "__main__":
    main()