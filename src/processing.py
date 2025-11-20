import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_union_features(features_path):
    """
    Carrega o arquivo UNION_features.tsv
    
    Args:
        features_path (str): Caminho para o arquivo UNION_features.tsv
        
    Returns:
        pd.DataFrame: DataFrame com as features dos genes
    """
    print("Carregando UNION_features.tsv")
    
    try:
        # Carrega o arquivo TSV
        features_df = pd.read_csv(features_path, sep='\t')
        print(f"Features carregadas: {features_df.shape[0]} genes x {features_df.shape[1]-1} features")
        
        # Remove a coluna 'gene' para análise das features
        gene_names = features_df['gene'].copy()
        features_only = features_df.drop('gene', axis=1)
        
        # Verifica valores faltantes
        missing_values = features_only.isnull().sum().sum()
        if missing_values > 0:
            print(f"Encontrados {missing_values} valores faltantes nas features")
            # Preenche valores faltantes com 0 (assumindo que são experimentos não realizados)
            features_only = features_only.fillna(0)
        
        # Informações sobre as features
        print(f"Tipos de dados das features: {features_only.dtypes.value_counts().to_dict()}")
        print(f"Range das features: [{features_only.min().min():.4f}, {features_only.max().max():.4f}]")
        
        return features_df, gene_names, features_only
        
    except Exception as e:
        print(f"Erro ao carregar features: {e}")
        return None, None, None


def load_union_labels(labels_path, classification_type='binary'):
    """
    Carrega o arquivo UNION_labels.tsv e processa os labels
    
    Args:
        labels_path (str): Caminho para o arquivo UNION_labels.tsv
        classification_type (str): Tipo de classificação - 'binary' (2class) ou 'multiclass' (3class)
        
    Returns:
        pd.DataFrame: DataFrame com os labels processados
    """
    print(f"Carregando UNION_labels.tsv (classificação: {classification_type})")
    
    try:
        # Carrega o arquivo TSV
        labels_df = pd.read_csv(labels_path, sep='\t')
        print(f"Labels carregados: {labels_df.shape[0]} genes")
        
        # Verifica se as colunas esperadas existem
        expected_columns = ['genes', '2class', '3class']
        if not all(col in labels_df.columns for col in expected_columns):
            print(f"ERRO: Colunas esperadas {expected_columns} não encontradas!")
            print(f"Colunas encontradas: {list(labels_df.columns)}")
            return None
        
        # Seleciona a coluna de classificação apropriada
        if classification_type == 'binary':
            label_column = '2class'
        elif classification_type == 'multiclass':
            label_column = '3class'
        else:
            print(f"ERRO: Tipo de classificação '{classification_type}' não reconhecido!")
            return None
        
        # Cria DataFrame com formato padrão (gene, label)
        labels_clean = pd.DataFrame({
            'gene': labels_df['genes'],
            'label': labels_df[label_column]
        })
        
        # Verifica valores únicos nos labels
        print(f"Valores únicos em {label_column}:", labels_clean['label'].value_counts(dropna=False))
        
        # Drop NaN values in label column
        initial_count = len(labels_clean)
        labels_clean = labels_clean.dropna(subset=['label'])
        print(f"Genes with NaN dropped: {initial_count - len(labels_clean)}")
        
        # Convert labels to integers
        labels_clean = labels_clean.copy()
        labels_clean.loc[:, 'label'] = labels_clean['label'].astype(int)
        
        # Verifica distribuição das classes
        class_distribution = labels_clean['label'].value_counts().sort_index()
        print("Distribuição das classes:")
        
        if classification_type == 'binary':
            print(f"  Classe 0 (passenger): {class_distribution.get(0, 0)} genes")
            print(f"  Classe 1 (cancer): {class_distribution.get(1, 0)} genes")
            
            if len(class_distribution) == 2:
                ratio = class_distribution[1] / class_distribution[0] if class_distribution.get(0, 0) > 0 else float('inf')
                print(f"  Razão cancer/passenger: {ratio:.4f}")
        else:  # multiclass
            print(f"  Classe 0 (passenger): {class_distribution.get(0, 0)} genes")
            print(f"  Classe 1 (TSG): {class_distribution.get(1, 0)} genes") 
            print(f"  Classe 2 (Oncogene): {class_distribution.get(2, 0)} genes")
        
        return labels_clean
        
    except Exception as e:
        print(f"Erro ao carregar labels: {e}")
        return None


def align_features_and_labels(features_df, labels_df):
    """
    Alinha as features e labels, mantendo apenas genes presentes em ambos
    
    Args:
        features_df (pd.DataFrame): DataFrame com features
        labels_df (pd.DataFrame): DataFrame com labels
        
    Returns:
        tuple: (X, y, gene_names) - features, labels e nomes dos genes alinhados
    """
    print("Alinhando features e labels")
    
    # Encontra genes comuns
    common_genes = set(features_df['gene']) & set(labels_df['gene'])
    print(f"Genes comuns entre features e labels: {len(common_genes)}")
    
    if len(common_genes) == 0:
        print("ERRO: Nenhum gene comum encontrado!")
        return None, None, None
    
    # Filtra datasets para genes comuns
    features_aligned = features_df[features_df['gene'].isin(common_genes)].copy()
    labels_aligned = labels_df[labels_df['gene'].isin(common_genes)].copy()
    
    # Ordena por gene para garantir alinhamento
    features_aligned = features_aligned.sort_values('gene').reset_index(drop=True)
    labels_aligned = labels_aligned.sort_values('gene').reset_index(drop=True)
    
    # Verifica se a ordem dos genes é idêntica
    if not features_aligned['gene'].equals(labels_aligned['gene']):
        print("ERRO: Genes não estão alinhados corretamente!")
        return None, None, None
    
    # Extrai features (X) e labels (y)
    X = features_aligned.drop('gene', axis=1).values
    y = labels_aligned['label'].values
    gene_names = features_aligned['gene'].values
    
    # Garante que y seja do tipo inteiro para np.bincount
    # Como os labels já foram processados na função load_union_labels,
    # apenas garantimos que sejam inteiros
    y = y.astype(int)
    
    print(f"Dataset final: {X.shape[0]} genes x {X.shape[1]} features")
    print(f"Distribuição final das classes: {np.bincount(y)}")
    
    return X, y, gene_names


def prepare_dataset(features_path, labels_path, classification_type='binary'):
    """
    Função principal para preparar o dataset completo
    
    Args:
        features_path (str): Caminho para UNION_features.tsv
        labels_path (str): Caminho para UNION_labels.tsv
        classification_type (str): Tipo de classificação - 'binary' ou 'multiclass'
        
    Returns:
        tuple: (X, y, gene_names, feature_names) - dataset completo preparado
    """
    print("="*60)
    print(f"PREPARANDO DATASET PARA CLASSIFICAÇÃO ({classification_type.upper()})")
    print("="*60)
    
    # Carrega features
    features_df, gene_names_feat, features_only = load_union_features(features_path)
    if features_df is None:
        return None, None, None, None

    print("-"*40)
    
    # Carrega labels
    labels_df = load_union_labels(labels_path, classification_type)
    if labels_df is None:
        return None, None, None, None
    
    print("-"*40)
    
    # Alinha features e labels
    X, y, gene_names = align_features_and_labels(features_df, labels_df)
    if X is None:
        return None, None, None, None
    
    # Nomes das features
    feature_names = features_df.drop('gene', axis=1).columns.tolist()
    
    print("-"*40)
    print("DATASET PREPARADO COM SUCESSO!")
    print(f"Shape final: X={X.shape}, y={y.shape}")
    print(f"Genes: {len(gene_names)}")
    print(f"Features: {len(feature_names)}")
    print("="*60)
    
    return X, y, gene_names


def prepare_renan_data(labels_path = "./renan/data_files/labels/UNION_labels.tsv", features_path = "./renan/data_files/omics_features/UNION_features.tsv"):
    """
    Prepara os dados do formato do Renan (labels True/False/NaN) para uso em modelagem.
    Args:
        labels_path (str): Caminho para renan/data_files/labels/UNION_labels.tsv
        features_path (str): Caminho para renan/data_files/omics_features/UNION_features.tsv
    Returns:
        tuple: (X, y, gene_names, feature_names) - dataset pronto para modelagem
    """
    print("Usando formato de labels do Renan (True/False/NaN)")
    # Carrega labels
    print(labels_path)
    labels_df = pd.read_csv(labels_path, sep='\t')
    print(f"Labels carregados: {labels_df.shape[0]} genes")
    print(labels_df.head())

    # Mantém apenas colunas relevantes
    if not set(['gene', 'label']).issubset(labels_df.columns):
        print(f"ERRO: Colunas esperadas ['gene', 'label'] não encontradas! Encontradas: {list(labels_df.columns)}")
        return None, None, None, None
    # Converte labels para 0/1, ignora NaN
    labels_df = labels_df.dropna(subset=['label'])
    labels_df['label'] = labels_df['label'].map({'True': 1, 'False': 0, True: 1, False: 0}).astype(int)
    print(f"Distribuição dos labels: {labels_df['label'].value_counts().to_dict()}")
    # Carrega features
    features_df = pd.read_csv(features_path, sep='\t')
    print(f"Features carregadas: {features_df.shape[0]} genes x {features_df.shape[1]-1} features")
    # Alinha genes
    common_genes = set(labels_df['gene']) & set(features_df['gene'])
    print(f"Genes comuns: {len(common_genes)}")
    if len(common_genes) == 0:
        print("ERRO: Nenhum gene comum encontrado!")
        return None, None, None, None
    labels_aligned = labels_df[labels_df['gene'].isin(common_genes)].sort_values('gene').reset_index(drop=True)
    features_aligned = features_df[features_df['gene'].isin(common_genes)].sort_values('gene').reset_index(drop=True)
    if not features_aligned['gene'].equals(labels_aligned['gene']):
        print("ERRO: Genes não estão alinhados corretamente!")
        return None, None, None, None
    X = features_aligned.drop('gene', axis=1).values
    y = labels_aligned['label'].values
    gene_names = features_aligned['gene'].values
    feature_names = features_aligned.drop('gene', axis=1).columns.tolist()
    print(f"Shape final: X={X.shape}, y={y.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Genes: {len(gene_names)}")
    return X, y, gene_names, feature_names