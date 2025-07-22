import pandas as pd

from collect_data import process_oncokb, process_ncg, process_cgc

ONCOKB_FILE = './data/onkoKB.tsv'
NCG_FILE = './data/NCG_cancerdrivers_annotation_supporting_evidence.tsv'
COSMIC_FILE = './data/cosmic.tsv'
OUTPUT_FILE = 'processed_genes_list.csv'

if __name__ == "__main__":
    # Processa cada arquivo de dados
    oncokb_genes = process_oncokb(ONCOKB_FILE)
    ncg_genes = process_ncg(NCG_FILE)
    cgc_genes = process_cgc(COSMIC_FILE)

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