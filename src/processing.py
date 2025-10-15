import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def load_union_features(path):
    """
    Carrega e processa o arquivo de features UNION_features.tsv
    """
    print("Carregando UNION_features.tsv...")

    try:
        df = pd.read_csv(path, sep='\t')
        print(f"Features carregadas: {df.shape[0]} genes x {df.shape[1] - 1} features")

        gene_names = df['gene'].copy()
        features = df.drop(columns='gene')

        missing = features.isnull().sum().sum()
        if missing:
            print(f"Encontrados {missing} valores faltantes. Preenchendo com 0.")
            features.fillna(0, inplace=True)

        print(f"Tipos de dados: {features.dtypes.value_counts().to_dict()}")
        print(f"Range das features: [{features.min().min():.4f}, {features.max().max():.4f}]")

        return df, gene_names, features

    except Exception as e:
        print(f"Erro ao carregar features: {e}")
        return None, None, None


def _convert_labels(label_series):
    """
    Converte diferentes tipos de labels para valores binários (0/1)
    """
    series = label_series.copy()

    if series.dtype == 'object':
        unique = set(str(v).lower() for v in series.unique())
        print(f"Valores únicos (normalizados): {unique}")

        def to_binary(val):
            val = str(val).lower()
            return int(val in ['true', '1', 'yes', 'positive'])

        return series.apply(to_binary)

    if series.dtype == 'bool':
        return series.astype(int)

    if np.issubdtype(series.dtype, np.integer):
        return series.astype(int)

    try:
        return series.astype(int)
    except Exception:
        print("Erro ao converter labels para inteiros. Verifique os dados.")
        raise


def load_union_labels(labels_path, classification_type='binary'):
    """
    Carrega o arquivo UNION_labels.tsv e processa os labels
    
    Args:
        labels_path (str): Caminho para o arquivo UNION_labels.tsv
        classification_type (str): Tipo de classificação - 'binary' (2class) ou 'multiclass' (3class)
        
    Returns:
        pd.DataFrame: DataFrame com os labels processados
    """
    print(f"Carregando UNION_labels.tsv (classificação: {classification_type})...")
    
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
        
        # Treat NaN values as class 0 (passenger genes) instead of removing them
        initial_count = len(labels_clean)
        labels_clean['label'] = labels_clean['label'].fillna(0.0)  # Fill NaN with 0
        
        print(f"Genes with NaN converted to class 0 (passenger): {initial_count - labels_clean['label'].sum()}")
        
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
    Alinha os dados de features e labels com base no nome do gene
    """
    print("Alinhando features e labels...")

    common_genes = set(features_df['gene']) & set(labels_df['gene'])
    print(f"Genes comuns: {len(common_genes)}")

    if not common_genes:
        print("Erro: Nenhum gene comum encontrado!")
        return None, None, None

    f_aligned = features_df[features_df['gene'].isin(common_genes)].copy()
    l_aligned = labels_df[labels_df['gene'].isin(common_genes)].copy()

    f_aligned.sort_values('gene', inplace=True)
    l_aligned.sort_values('gene', inplace=True)

    f_aligned.reset_index(drop=True, inplace=True)
    l_aligned.reset_index(drop=True, inplace=True)

    if not f_aligned['gene'].equals(l_aligned['gene']):
        print("Erro: Ordem dos genes não coincide!")
        return None, None, None

    X = f_aligned.drop(columns='gene').values
    y = l_aligned['label'].astype(int).values
    gene_names = f_aligned['gene'].values

    print(f"Dataset final: {X.shape[0]} genes x {X.shape[1]} features")
    print(f"Distribuição das classes: {np.bincount(y)}")

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
    
    return X, y, gene_names, feature_names


def get_dataset_info(X, y, gene_names, feature_names):
    """
    Retorna um resumo estatístico do dataset
    """
    return {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_genes': len(gene_names),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'feature_stats': {
            'mean': np.mean(X),
            'std': np.std(X),
            'min': np.min(X),
            'max': np.max(X),
            'zeros_percentage': (X == 0).mean() * 100,
        }
    }


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Divide o dataset em treino e teste, mantendo a proporção de classes
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Dataset dividido:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste:  {X_test.shape[0]} amostras")
    print(f"  Classes treino: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Classes teste:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test
