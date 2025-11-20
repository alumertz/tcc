CPDB_PATHWAYS_FILE = './data/CPDB_pathways_genes.tsv'
ONCOKB_FILE = './data/onkoKB.tsv'
NCG_FILE = './data/NCG_cancerdrivers_annotation_supporting_evidence.tsv'
NCG_CANDIDATE_HEALTHY_FILE = './data/NCG_candidate_and_remaining_healthy_drivers.txt'
NCG_CANONICAL_HEALTHY_FILE = './data/NCG_cancer_and_healthy_drivers.txt'
COSMIC_FILE = './data/Cosmic_11_2025.tsv'
OMIM_FILE = './data/All_Diseases_OMIM.txt'
HGNC_FILE = './data/hgnc-symbol-check.csv'
BUSHMAN_FILE = './data/Bushman_group_allOnco.tsv'
UNION_FEATURES_FILE = './data/UNION_features.tsv'
OUTPUT_DIR = './data/processed'

def load_hgnc_mapping(file_path=HGNC_FILE):
    """
    Carrega o arquivo HGNC de verificação de símbolos e cria um dicionário de mapeamento.
    Retorna um dicionário que mapeia símbolos de entrada para símbolos aprovados.
    """
    print("Carregando mapeamento HGNC...")
    try:
        df = pd.read_csv(file_path)
        
        # Criar dicionário de mapeamento
        mapping_dict = {}
        unmatched_genes = []
        withdrawn_genes = []
        
        for _, row in df.iterrows():
            input_symbol = row['Input']
            match_type = row['Match type']
            approved_symbol = row['Approved symbol']
            
            if match_type in ['Approved symbol', 'Alias symbol', 'Previous symbol']:
                mapping_dict[input_symbol] = approved_symbol
            elif match_type == 'Unmatched':
                unmatched_genes.append({
                    'input': input_symbol,
                    'match_type': match_type,
                    'approved_symbol': approved_symbol if pd.notna(approved_symbol) else 'N/A'
                })
            elif match_type == 'Entry withdrawn':
                withdrawn_genes.append({
                    'input': input_symbol,
                    'match_type': match_type,
                    'approved_symbol': approved_symbol if pd.notna(approved_symbol) else 'N/A'
                })
        
        print(f"HGNC: Carregados {len(mapping_dict)} mapeamentos de símbolos")
        print(f"HGNC: {len(unmatched_genes)} genes não encontrados")
        print(f"HGNC: {len(withdrawn_genes)} genes retirados")
        
        # Imprime informações sobre genes não aprovados
        if unmatched_genes:
            print("\n=== GENES NÃO ENCONTRADOS (UNMATCHED) ===")
            for gene in unmatched_genes:
                print(f"Input: {gene['input']}, Tipo: {gene['match_type']}, Aprovado: {gene['approved_symbol']}")
        
        if withdrawn_genes:
            print("\n=== GENES RETIRADOS (WITHDRAWN) ===")
            for gene in withdrawn_genes:
                print(f"Input: {gene['input']}, Tipo: {gene['match_type']}, Aprovado: {gene['approved_symbol']}")
        
        return mapping_dict, unmatched_genes, withdrawn_genes
        
    except FileNotFoundError:
        print(f"Erro: O arquivo HGNC '{file_path}' não foi encontrado.")
        return {}, [], []

def apply_hgnc_mapping(df, hgnc_mapping, column_name='symbol'):
    """
    Aplica o mapeamento HGNC aos símbolos de genes em um DataFrame.
    Remove genes não encontrados ou retirados.
    
    Returns:
    --------
    tuple: (df_mapped, removed_genes_set)
    """
    original_count = len(df)
    df_mapped = df.copy()
    
    # Mapeia os símbolos
    df_mapped[column_name] = df_mapped[column_name].map(hgnc_mapping).fillna(df_mapped[column_name])
    
    # Remove genes que não puderam ser mapeados (não estão no dicionário de mapeamento)
    # Isso remove genes "Unmatched" e "Entry withdrawn"
    unmapped_genes = df_mapped[~df_mapped[column_name].isin(hgnc_mapping.values()) & 
                              ~df_mapped[column_name].isin(hgnc_mapping.keys())][column_name].tolist()
    
    if unmapped_genes:
        print(f"Removendo {len(unmapped_genes)} genes não mapeados: {unmapped_genes}")
    
    # Mantém apenas genes que estão mapeados ou já são símbolos aprovados
    df_mapped = df_mapped[df_mapped[column_name].isin(hgnc_mapping.values()) | 
                         df_mapped[column_name].isin(hgnc_mapping.keys())]
    
    # Remove duplicatas que podem ter surgido do mapeamento
    df_mapped = df_mapped.drop_duplicates(subset=[column_name]).reset_index(drop=True)
    
    final_count = len(df_mapped)
    print(f"Mapeamento aplicado: {original_count} → {final_count} genes (removidos: {original_count - final_count})")
    
    return df_mapped, set(unmapped_genes)

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

                # features_genes já é passado como argumento, não precisa ler do arquivo
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

def process_NCG_healthy(mode, hgnc_mapping):
    """
    Processa genes healthy drivers para modo 'canonical' ou 'candidate'.
    Para 'canonical', lê NCG_cancer_and_healthy_drivers.txt e marca todos como TSG.
    Para 'candidate', lê NCG_candidate_and_remaining_healthy_drivers.txt e marca todos como TSG.
    Retorna DataFrame com colunas: symbol, Oncogene, TSG
    """
    if mode == 'canonical':
        file_path = NCG_CANONICAL_HEALTHY_FILE
    elif mode == 'candidate':
        file_path = NCG_CANDIDATE_HEALTHY_FILE
    else:
        raise ValueError("Modo inválido para process_NCG_healthy: use 'canonical' ou 'candidate'.")

    try:
        df = pd.read_csv(file_path, sep='\t', header=None)
        # Assume primeira coluna como símbolo do gene
        df.columns = ['symbol']
        df['Oncogene'] = 'No'
        df['TSG'] = 'Yes'
        df = df[['symbol', 'Oncogene', 'TSG']]
        print(f"NCG Healthy ({mode}): Encontrados {len(df)} genes TSG.")
        df_mapped, removed = apply_hgnc_mapping(df, hgnc_mapping)
        return df_mapped, removed
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG']), set()

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

def process_bushman_candidates(hgnc_mapping, file_path=BUSHMAN_FILE):
    """
    Carrega e processa a lista de genes candidatos do arquivo Bushman_group_allOnco.tsv.
    Aplica mapeamento HGNC.
    Returns:
    --------
    tuple: (bushman_genes_df, removed_genes_set)
    """
    print("Processando Bushman_group_allOnco.tsv...")
    try:
        df_raw = pd.read_csv(file_path, sep='\t', header=None)
        # Remove header if present (first row contains non-gene text)
        candidate_symbols = []
        for i, row in df_raw.iterrows():
            # Skip header or malformed rows
            if i == 0 and not str(row[1]).isalpha():
                continue
            symbol = str(row[1]).strip()
            synonyms = str(row[2]).strip() if len(row) > 2 else ''
            mapped = False
            # Try mapping symbol
            if symbol in hgnc_mapping:
                candidate_symbols.append(hgnc_mapping[symbol])
                mapped = True
            
            # If still not mapped, keep original symbol for reporting
            if not mapped and symbol and symbol.isalpha():
                candidate_symbols.append(symbol)
        df = pd.DataFrame({'symbol': candidate_symbols})
        df['Oncogene'] = 'No'
        df['TSG'] = 'No'
        df = df[['symbol', 'Oncogene', 'TSG']]
        print(f"Bushman: Encontrados {len(df)} genes candidatos.")
        df_mapped, removed = apply_hgnc_mapping(df, hgnc_mapping)
        return df_mapped, removed
    except Exception as e:
        print(f"Erro ao processar Bushman_group_allOnco.tsv: {e}")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG']), set()

def process_cpdb_cancer_pathways(file_path=CPDB_PATHWAYS_FILE):
    """
    Processa CPDB_pathways_genes.tsv, filtrando apenas pathways que incluem 'cancer'.
    Extrai genes da coluna 'hgnc_symbol_ids' e retorna DataFrame padrão (symbol, Oncogene, TSG).
    """
    print("Processando CPDB_pathways_genes.tsv para pathways de câncer...")
    try:
        df = pd.read_csv(file_path, sep='\t')
        # Filtra pathways que incluem 'cancer' (case-insensitive)
        df_cancer = df[df['pathway'].str.contains('cancer', case=False, na=False)]
        # Extrai todos os genes da coluna hgnc_symbol_ids (assume separados por vírgula ou espaço)
        gene_set = set()
        for genes_str in df_cancer['hgnc_symbol_ids']:
            # Suporta separação por vírgula ou espaço
            genes = [g.strip() for g in str(genes_str).replace(',', ' ').split() if g.strip()]
            gene_set.update(genes)
        print(f"CPDB: Encontrados {len(gene_set)} genes relacionados a pathways de câncer.")
        # Cria DataFrame padrão
        df_genes = pd.DataFrame({'symbol': list(gene_set)})
        df_genes['Oncogene'] = 'No'
        df_genes['TSG'] = 'No'
        df_genes = df_genes[['symbol', 'Oncogene', 'TSG']]
        return df_genes
    except Exception as e:
        print(f"Erro ao processar CPDB_pathways_genes.tsv: {e}")
        return pd.DataFrame(columns=['symbol', 'Oncogene', 'TSG']), set()
import pandas as pd
import numpy as np
import os

def get_canonical_genes():
    """
    Processa dados para obter genes canônicos (oncogenes e TSGs conhecidos).
    
    Returns:
    --------
    tuple: (canonical_genes_df, removed_genes_set)
    """
    print("\n=== PROCESSANDO GENES CANÔNICOS ===")
    
    # Carrega mapeamento HGNC
    hgnc_mapping, unmatched_genes, withdrawn_genes = load_hgnc_mapping()
    
    # Processa cada fonte de dados
    oncokb_genes = process_oncokb('canonical')
    ncg_genes = process_ncg('canonical')
    cgc_genes = process_cgc(tier=1)

    # Aplica mapeamento HGNC a cada fonte
    print("\nAplicando mapeamento HGNC aos genes OncoKB...")
    oncokb_genes, oncokb_removed = apply_hgnc_mapping(oncokb_genes, hgnc_mapping)
    
    print("Aplicando mapeamento HGNC aos genes NCG...")
    ncg_genes, ncg_removed = apply_hgnc_mapping(ncg_genes, hgnc_mapping)
    
    print("Aplicando mapeamento HGNC aos genes CGC...")
    cgc_genes, cgc_removed = apply_hgnc_mapping(cgc_genes, hgnc_mapping)
    
    # Combina genes removidos
    all_removed = oncokb_removed | ncg_removed | cgc_removed

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
    
    return final_genes_sorted, all_removed

def get_candidate_genes():
    """
    Processa dados para obter genes candidatos.
    
    Returns:
    --------
    tuple: (candidate_genes_df, removed_genes_set)
    """
    print("\n=== PROCESSANDO GENES CANDIDATOS ===")
    
    # Carrega mapeamento HGNC
    hgnc_mapping, unmatched_genes, withdrawn_genes = load_hgnc_mapping()
    

    # Processa cada fonte de dados
    cgc_genes = process_cgc(tier=2)
    oncokb_genes = process_oncokb('candidate')
    ncg_genes = process_ncg('candidate')
    omim_genes = process_omim()
    bushman_genes, bushman_removed = process_bushman_candidates(hgnc_mapping)
    cpdb_genes = process_cpdb_cancer_pathways()

    # Aplica mapeamento HGNC a cada fonte
    print("\nAplicando mapeamento HGNC aos genes CGC Tier 2...")
    cgc_genes, cgc_removed = apply_hgnc_mapping(cgc_genes, hgnc_mapping)

    print("Aplicando mapeamento HGNC aos genes OncoKB candidatos...")
    oncokb_genes, oncokb_removed = apply_hgnc_mapping(oncokb_genes, hgnc_mapping)

    print("Aplicando mapeamento HGNC aos genes NCG candidatos...")
    ncg_genes, ncg_removed = apply_hgnc_mapping(ncg_genes, hgnc_mapping)

    print("Aplicando mapeamento HGNC aos genes OMIM...")
    omim_genes, omim_removed = apply_hgnc_mapping(omim_genes, hgnc_mapping)

    # Combina genes removidos
    all_removed = cgc_removed | oncokb_removed | ncg_removed | omim_removed | bushman_removed

    # Combina as listas
    print("\nCombinando as listas de genes candidatos...")
    combined_df = pd.concat([
        cgc_genes, oncokb_genes, ncg_genes, omim_genes, bushman_genes, cpdb_genes
    ], ignore_index=True)
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
    
    return final_genes_sorted, all_removed

def create_union_labels(canonical_genes, candidate_genes, features):
    """
    Cria arquivo UNION_labels.tsv com 3 colunas:
    - genes: símbolo do gene
    - 2class: classe binária (cancer=1, candidato=NaN, passenger=0)
    - 3class: 3classe (TSG=1, OC=2, candidato=NaN, passenger=0)
    """
    print("\n=== CRIANDO UNION_LABELS ===")
    
    # Identifica categorias dos genes
    canonical_symbols = set(canonical_genes['symbol'])
    candidate_symbols = set(candidate_genes['symbol'])
    features_genes = set(features.index.tolist())

    # Cria dicionário para lookup rápido de Oncogene/TSG
    canonical_info = canonical_genes.set_index('symbol').to_dict('index')
    candidate_info = candidate_genes.set_index('symbol').to_dict('index')

    # Lista de todos genes a considerar: união de features, canônicos e candidatos
    all_genes_set = features_genes | canonical_symbols | candidate_symbols
    all_genes_sorted = sorted(all_genes_set)

    genes = []
    binary_class = []
    multiclass_labels = []

    for symbol in all_genes_sorted:
        genes.append(symbol)
        # Binária: cancer = 1 (canônico), candidato = NaN, passenger = 0
        if symbol in canonical_symbols:
            binary_class.append(1)
        elif symbol in candidate_symbols:
            binary_class.append(np.nan)
        else:
            binary_class.append(0)
        # Multiclasse: TSG=1, OC=2, candidato=NaN, passenger=0
        if symbol in canonical_symbols:
            info = canonical_info.get(symbol, {})
            tsg = info.get('TSG', 'No')
            oncogene = info.get('Oncogene', 'No')
            if tsg == 'Yes':
                multiclass_labels.append(1)
            elif oncogene == 'Yes':
                multiclass_labels.append(2)
            else:
                multiclass_labels.append(0)
        elif symbol in candidate_symbols:
            multiclass_labels.append(np.nan)
        else:
            multiclass_labels.append(0)

    # Cria DataFrame final
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
    
    # Carrega mapeamento HGNC uma vez para uso em features
    hgnc_mapping, unmatched_genes, withdrawn_genes = load_hgnc_mapping()
    
    # Processa genes canônicos e candidatos
    canonical_genes, canonical_removed = get_canonical_genes()
    candidate_genes, candidate_removed = get_candidate_genes()
    
    # Combina todos os genes removidos
    all_removed_genes = canonical_removed | candidate_removed

    # Processa UNION_features.tsv
    #processed_features = process_union_features(hgnc_mapping, all_removed_genes)
    features = pd.read_csv(UNION_FEATURES_FILE, sep='\t', index_col=0)

    # Cria arquivo UNION_labels
    union_labels = create_union_labels(canonical_genes, candidate_genes, features)
    
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