import os


'''
    The code below scans multiple feature files to identify the presence of a specified list of additional genes in Ana’s labels.
'''



# List of genes to search for
genes_to_search = [
    "ABCD1P4", "BTF3P11", "C1orf141", "CENPCP1", "CYCSP42", "CYCSP5", "DHFRP2", "ERLN", "FAM47C", "GAPDHP26",
    "GBA1", "H3P6", "HADHAP1", "HCG22", "HCG9", "HLA-DRB6", "HLA-DRB9", "HLA-S", "HLA-X", "HMGN2P46",
    "IGH", "IGK", "KCP", "LINC00598", "LINC01235", "LINC01548", "LINC03040", "LIPI", "LORICRIN", "MAFIP",
    "MALRD1", "MRPL49P2", "MT-TP", "NBEAP1", "NHERF1", "NRAD1", "NUTM2D", "OR7H2P", "PABPC1P9", "PIERCE2",
    "POLR1HASP", "RHEBP1", "RNF217-AS1", "RNU7-2P", "RNU7-31P", "RNU7-34P", "RPL13AP14", "RPL23AP1", "RPL23AP12",
    "RPL23P4", "RPL28P4", "RPL30P9", "RPL31P10", "RPL31P13", "RPL35AP15", "RPL35P3", "RPL37AP1", "RPL39P28",
    "RPL3P11", "RPL3P2", "RPL6P5", "RPS14P1", "RPS20P32", "RPS23P3", "RPS3AP42", "RPS3AP46", "SLC67A1",
    "SNORA47", "SPATA31H1", "ST13P1", "ST13P12", "SUMO2P1", "TCEA1P2", "TRD-GTC9-1", "TRG", "WASF5P", "WHR1", "XBP1P1"
]

# List of file paths to search in
files = [
    "/home/kamille/Documentos/tcc/renan/data_files/omics_features/CPDB_features.tsv",
    "/home/kamille/Documentos/tcc/renan/data_files/omics_features/HPRD_features.tsv",
    "/home/kamille/Documentos/tcc/renan/data_files/omics_features/IREF_features.tsv",
    "/home/kamille/Documentos/tcc/renan/data_files/omics_features/MULTINET_features.tsv",
    "/home/kamille/Documentos/tcc/renan/data_files/omics_features/PCNET_features.tsv",
    "/home/kamille/Documentos/tcc/renan/data_files/omics_features/STRING_features.tsv"
]

# Function to search for genes in a file
def search_genes_in_file(file_path, genes):
    found_genes = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    gene = parts[0].strip()
                    if gene in genes:
                        found_genes.append(gene)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return found_genes

# Search in each file
for file_path in files:
    found = search_genes_in_file(file_path, genes_to_search)
    if found:
        for gene in found:
            print(f"Gene {gene} found in {os.path.basename(file_path)}")
    else:
        print(f"No matching genes found in {os.path.basename(file_path)}")