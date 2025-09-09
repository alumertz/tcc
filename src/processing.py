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
    print("Carregando UNION_features.tsv...")
    
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


def load_union_labels(labels_path):
    """
    Carrega o arquivo UNION_labels.tsv e processa os labels
    
    Args:
        labels_path (str): Caminho para o arquivo UNION_labels.tsv
        
    Returns:
        pd.DataFrame: DataFrame com os labels processados
    """
    print("Carregando UNION_labels.tsv...")
    
    try:
        # Carrega o arquivo TSV
        labels_df = pd.read_csv(labels_path, sep='\t')
        print(f"Labels carregados: {labels_df.shape[0]} genes")
        
        # Verifica valores únicos nos labels
        print("Valores únicos nos labels:", labels_df['label'].value_counts(dropna=False))
        
        # Remove genes sem label (valores NaN/vazios)
        labels_clean = labels_df.dropna(subset=['label'])
        removed_genes = len(labels_df) - len(labels_clean)
        if removed_genes > 0:
            print(f"Removidos {removed_genes} genes sem label")
        
        # Converte labels para formato binário (True/False -> 1/0)
        labels_clean = labels_clean.copy()
        if labels_clean['label'].dtype == 'object':
            # Se os labels são strings, tenta diferentes formatos
            unique_values = set(str(v).lower() for v in labels_clean['label'].unique())
            print(f"Valores únicos encontrados (lowercase): {unique_values}")
            
            if unique_values <= {'true', 'false'}:
                # Strings 'True'/'False' (case insensitive)
                labels_clean.loc[:, 'label'] = labels_clean['label'].apply(
                    lambda x: 1 if str(x).lower() == 'true' else 0
                )
            elif unique_values <= {'1', '0'}:
                # Strings '1'/'0'
                labels_clean.loc[:, 'label'] = labels_clean['label'].astype(int)
            else:
                print(f"AVISO: Valores de label não reconhecidos: {unique_values}")
                # Tenta conversão genérica
                labels_clean.loc[:, 'label'] = labels_clean['label'].apply(
                    lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'positive'] else 0
                )
        elif labels_clean['label'].dtype == 'bool':
            # Se os labels já são booleanos
            labels_clean.loc[:, 'label'] = labels_clean['label'].astype(int)
        elif labels_clean['label'].dtype in ['int64', 'int32', 'int']:
            # Se já são inteiros, não faz nada
            pass
        else:
            # Conversão forçada para inteiro
            labels_clean.loc[:, 'label'] = labels_clean['label'].astype(int)
        
        # Verifica distribuição das classes
        class_distribution = labels_clean['label'].value_counts()
        print("Distribuição das classes:")
        print(f"  Classe 0 (não-alvo): {class_distribution.get(0, 0)} genes")
        print(f"  Classe 1 (alvo): {class_distribution.get(1, 0)} genes")
        
        if len(class_distribution) == 2:
            ratio = class_distribution[1] / class_distribution[0]
            print(f"  Razão positivos/negativos: {ratio:.4f}")
        
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
    print("Alinhando features e labels...")
    
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


def prepare_dataset(features_path, labels_path):
    """
    Função principal para preparar o dataset completo
    
    Args:
        features_path (str): Caminho para UNION_features.tsv
        labels_path (str): Caminho para UNION_labels.tsv
        
    Returns:
        tuple: (X, y, gene_names, feature_names) - dataset completo preparado
    """
    print("="*60)
    print("PREPARANDO DATASET PARA CLASSIFICAÇÃO")
    print("="*60)
    
    # Carrega features
    features_df, gene_names_feat, features_only = load_union_features(features_path)
    if features_df is None:
        return None, None, None, None
    
    print("-"*40)
    
    # Carrega labels
    labels_df = load_union_labels(labels_path)
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
    
    return X, y, gene_names, feature_names


def get_dataset_info(X, y, gene_names, feature_names):
    """
    Retorna informações detalhadas sobre o dataset
    """
    info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_genes': len(gene_names),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'feature_stats': {
            'mean': np.mean(X),
            'std': np.std(X),
            'min': np.min(X),
            'max': np.max(X),
            'zeros_percentage': np.mean(X == 0) * 100
        }
    }
    
    return info


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Divide o dataset em treino e teste mantendo a proporção das classes
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"Dataset dividido:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    print(f"  Distribuição treino: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Distribuição teste: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_train, X_test, y_train, y_test
