#!/usr/bin/env python3
"""
Arquivo principal para experimentação com modelos de classificação
para predição de genes-alvo usando dados ômicos.
"""

import sys
import os
import argparse
sys.path.append('/Users/i583975/git/tcc')

from src.processing import prepare_dataset, prepare_renan_data
from src.models import (
    optimize_decision_tree_classifier,
    optimize_random_forest_classifier,
    optimize_gradient_boosting_classifier,
    optimize_hist_gradient_boosting_classifier,
    optimize_knn_classifier,
    optimize_mlp_classifier,
    optimize_svc_classifier,
    optimize_catboost_classifier,
    optimize_xgboost_classifier
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from evaluation import evaluate_model_default
from src.reports import generate_experiment_folder_name
import warnings
warnings.filterwarnings('ignore')

N_TRIALS = 30

def get_data_paths(use_renan=False):
    """
    Retorna os caminhos dos arquivos baseado na fonte de dados escolhida
    
    Args:
        use_renan (bool): Se True, usa arquivos do Renan; se False, usa arquivos da Ana
        
    Returns:
        tuple: (features_path, labels_path, data_source)
    """
    if use_renan:
        features_path = "./renan/data_files/omics_features/UNION_features.tsv"
        labels_path = "./renan/data_files/labels/UNION_labels.tsv"
        data_source = "RENAN"
        print(f" Usando dados do RENAN:")
        print(f"   Features: renan/data_files/omics_features/UNION_features.tsv")
        print(f"   Labels: renan/data_files/labels/UNION_labels.tsv")
        print(f"   Formato labels: gene, label (True/False/NaN)")
    else:
        features_path = "./data/UNION_features.tsv"
        labels_path = "./data/processed/UNION_labels.tsv"
        data_source = "ANA"
        print(f" Usando dados da ANA:")
        print(f"   Features: data/UNION_features.tsv")
        print(f"   Labels: data/processed/UNION_labels.tsv") 
        print(f"   Formato labels: genes, 2class (binary), 3class (multiclass)")
    print()
    
    return features_path, labels_path, data_source

def parse_arguments():
    """
    Processa argumentos da linha de comando
    
    Returns:
        argparse.Namespace: Argumentos processados
    """
    parser = argparse.ArgumentParser(
        description='Experimentação com modelos de classificação para predição de genes-alvo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py                    # Modelos otimizados, dados Ana, classificação binária
  python main.py -default           # Modelos com parâmetros padrão (rápido)
  python main.py -lessparams        # Otimização com conjunto reduzido de parâmetros (mais rápido)
  python main.py -renan             # Usa arquivos do Renan (formato original)
  python main.py -multiclass        # Classificação multiclasse (TSG vs Oncogene vs Passenger)
  python main.py -default -multiclass # Parâmetros padrão + multiclasse
  python main.py -balancedata smoteenn # Com balanceamento SMOTEENN
  python main.py -default -balancedata adasyn # Parâmetros padrão + balanceamento ADASYN
  python main.py -multiclass -balancedata kmeanssmote # Multiclasse + KMeansSMOTE
  python main.py -lessparams -model catboost # Otimização rápida apenas CatBoost
  python main.py --help             # Mostra esta ajuda
        """
    )
    
    parser.add_argument(
        '-renan', '--renan', 
        action='store_true',
        help='Usa arquivos de dados do Renan (formato original: gene, label)'
    )
    
    parser.add_argument(
        '-multiclass', '--multiclass',
        action='store_true',
        help='Usa classificação multiclasse (TSG=1, Oncogene=2) ao invés de binária (cancer=1)'
    )
    
    parser.add_argument(
        '-default', '--default',
        action='store_true',
        help='Executa modelos com parâmetros padrão (sem otimização Optuna)'
    )
    
    parser.add_argument(
        '-balancedata', '--balancedata',
        type=str,
        choices=['none', 'smoteenn', 'smotetomek', 'randomundersampler', 'tomeklinks', 'smoten', 'adasyn', 'kmeanssmote'],
        default='none',
        help='Estratégia de balanceamento de dados: none (sem balanceamento), smoteenn, smotetomek, randomundersampler, tomeklinks, smoten, adasyn, kmeanssmote'
    )
    parser.add_argument(
        '-model', '--model',
        type=str,
        nargs='+',
        choices=['catboost', 'decisiontree', 'gradientboosting', 'histgradientboosting', 'knn', 'mlp', 'randomforest', 'svc', 'xgboost'],
        default=None,
        help='Executa um ou mais modelos específicos (ex: -model randomforest knn mlp)'
    )
    parser.add_argument(
        '-lessparams', '--lessparams',
        action='store_true',
        help='Usa conjunto reduzido de parâmetros para otimização (busca mais rápida)'
    )
    
    return parser.parse_args()

def run_single_model_optimize(model_name, optimizer_func, X, y, n_trials=10, data_source="ana", classification_type="binary", use_less_params=False):
    """
    Executa um único modelo de classificação
    
    Args:
        model_name (str): Nome do modelo
        optimizer_func (function): Função de otimização do modelo
        X (np.array): Features
        y (np.array): Labels
        n_trials (int): Número de trials para otimização
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
        use_less_params (bool): Se True, usa conjunto reduzido de parâmetros
        
    Returns:
        dict: Resultados do modelo
    """
    print("="*80)
    print(f"EXECUTANDO MODELO: {model_name}")
    print("="*80)
    
    try:
        # Executa otimização (com salvamento automático)
        best_model, test_metrics = optimizer_func(X, y, n_trials=n_trials, save_results=True, 
                                                data_source=data_source, classification_type=classification_type,
                                                use_nested_cv=True, outer_cv_folds=5, use_less_params=use_less_params)
        
        results = {
            'model_name': model_name,
            'status': 'success',
            'model': best_model,
            'test_metrics': test_metrics
        }
        
        print(f"✓ {model_name} executado com sucesso!")
        return results
        
    except Exception as e:
        print(f"✗ Erro ao executar {model_name}: {e}")
        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e),
            'model': None
        }

def run_all_models_optimize(X, y, n_trials=10, data_source="ana", classification_type="binary", use_less_params=False):
    """
    Executa todos os modelos de classificação
    
    Args:
        X (np.array): Features
        y (np.array): Labels
        n_trials (int): Número de trials para otimização
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
        use_less_params (bool): Se True, usa conjunto reduzido de parâmetros
        
    Returns:
        list: Lista com resultados de todos os modelos
    """
    # Definição dos modelos e suas funções de otimização
    models_config = [
        ("Decision Tree", optimize_decision_tree_classifier),
        ("K-Nearest Neighbors", optimize_knn_classifier),
        ("Support Vector Classifier", optimize_svc_classifier),
        ("Random Forest", optimize_random_forest_classifier),
        ("Gradient Boosting", optimize_gradient_boosting_classifier),
        ("Histogram Gradient Boosting", optimize_hist_gradient_boosting_classifier),
        ("Multi-Layer Perceptron", optimize_mlp_classifier),
        ("XGBoost", optimize_xgboost_classifier),
        ("CatBoost", optimize_catboost_classifier)
    ]
    
    results = []
    
    print("INICIANDO EXPERIMENTAÇÃO COM TODOS OS MODELOS")
    print(f"Dataset: {X.shape[0]} amostras x {X.shape[1]} features")
    print(f"Número de trials por modelo: {n_trials}")
    print(f"Conjunto de parâmetros: {'Reduzido' if use_less_params else 'Completo'}")
    
    
    for i, (model_name, optimizer_func) in enumerate(models_config, 1):
        print(f"Progresso: {i}/{len(models_config)} modelos")
        
        if model_name == "Support Vector Classifier":
            n_trials = 5
        else:
            n_trials = N_TRIALS
        
        result = run_single_model_optimize(model_name, optimizer_func, X, y, n_trials, data_source, classification_type, use_less_params)
        results.append(result)
        
    return results

def run_all_default_models(X, y, data_source="ana", classification_type="binary", balance_strategy=None):
    """
    Executa todos os modelos com parâmetros padrão
    Pipeline unificado para todos: StandardScaler + Classifier
    
    Args:
        X (np.array): Features
        y (np.array): Labels
        data_source (str): "ana" ou "renan"
        classification_type (str): "binary" ou "multiclass"
        balance_strategy (str): Estratégia de balanceamento de dados
        
    Returns:
        list: Lista com resultados de todos os modelos
    """
    # Modelos com parâmetros padrão (mesma lista do main_default_models.py)
    default_models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ("Histogram Gradient Boosting", HistGradientBoostingClassifier(random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Multi-Layer Perceptron", MLPClassifier(random_state=42, max_iter=1000)),
        ("Support Vector Classifier", SVC(probability=True, random_state=42)),
        ("CatBoost", CatBoostClassifier(random_state=42, verbose=False)),
        ("XGBoost", XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss'))
    ]
    
    results = []
    
    print("INICIANDO AVALIAÇÃO COM PARÂMETROS PADRÃO")
    print(f"Dataset: {X.shape[0]} amostras x {X.shape[1]} features")
    print(f"Pipeline: StandardScaler + Classifier (unificado)")

    experiment_folder = generate_experiment_folder_name(data_source, "default", classification_type)
    experiment_folder = experiment_folder + "_"+ balance_strategy if balance_strategy else experiment_folder
    
    for i, (model_name, model) in enumerate(default_models, 1):
        print(f"\nProgresso: {i}/{len(default_models)} modelos")
        experiment_dir = os.path.join("./results", experiment_folder)
        os.makedirs(experiment_dir, exist_ok=True)
        try:
            result = evaluate_model_default(model, model_name, X, y, experiment_dir, classification_type, balance_strategy)
            results.append(result)
            print(f"✓ {model_name} executado com sucesso!")
        except Exception as e:
            print(f"✗ Erro ao executar {model_name}: {e}")
            results.append({
                'model_name': model_name,
                'status': 'error',
                'error': str(e)
            })
            
    
    # Save summary file with all metrics for all algorithms
    from src.reports import save_default_experiment_summary
    save_default_experiment_summary(experiment_dir, results, balance_strategy)
    return results

def main(use_renan=False, use_multiclass=False, use_default=False, balance_strategy='none', model_names=None, use_less_params=False):
    """
    Função principal do experimento
    
    Args:
        use_renan (bool): Se True, usa arquivos do Renan; se False, usa arquivos da Ana
        use_multiclass (bool): Se True, usa classificação multiclasse; se False, usa binária
        use_default (bool): Se True, usa parâmetros padrão; se False, otimiza com Optuna
        balance_strategy (str): Estratégia de balanceamento de dados
        model_names (list): Lista de nomes dos modelos a executar (None para todos)
        use_less_params (bool): Se True, usa conjunto reduzido de parâmetros
    """
    print("CLASSIFICAÇÃO DE GENES-ALVO USANDO DADOS ÔMICOS")
    print("="*80)
    
    # Inicializar timestamp do experimento para toda a sessão
    from src.reports import set_experiment_timestamp
    experiment_timestamp = set_experiment_timestamp()
    print(f"Timestamp do experimento: {experiment_timestamp}")
    
    # Obtém os caminhos dos arquivos baseado na fonte escolhida
    features_path, labels_path, data_source = get_data_paths(use_renan)
    
    # Verifica se os arquivos existem
    if not os.path.exists(features_path):
        print(f"Arquivo de features não encontrado: {features_path}")
        return
    
    if not os.path.exists(labels_path):
        print(f"Arquivo de labels não encontrado: {labels_path}")
        return
    
    print("Arquivos encontrados com sucesso!")
    
    # Prepara o dataset
    print("Carregando e preparando dados...")
    if use_renan:
        print(" Usando formato de labels do Renan (True/False/NaN)")
        X, y, gene_names, feature_names = prepare_renan_data()
        
    else:
        classification_type = 'multiclass' if use_multiclass else 'binary'
        X, y, gene_names = prepare_dataset(features_path, labels_path, classification_type)
    
    if X is None:
        print("Erro ao preparar dataset. Abortando.")
        return
    # Dicionário de modelos e funções de otimização
    model_map = {
        'decisiontree': ("Decision Tree", optimize_decision_tree_classifier),
        'knn': ("K-Nearest Neighbors", optimize_knn_classifier),
        'svc': ("Support Vector Classifier", optimize_svc_classifier),
        'randomforest': ("Random Forest", optimize_random_forest_classifier),
        'gradientboosting': ("Gradient Boosting", optimize_gradient_boosting_classifier),
        'histgradientboosting': ("Histogram Gradient Boosting", optimize_hist_gradient_boosting_classifier),
        'mlp': ("Multi-Layer Perceptron", optimize_mlp_classifier),
        'xgboost': ("XGBoost", optimize_xgboost_classifier),
        'catboost': ("CatBoost", optimize_catboost_classifier)
    }
    
    # Dicionário de modelos padrão
    default_model_map = {
        'decisiontree': ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        'knn': ("K-Nearest Neighbors", KNeighborsClassifier()),
        'svc': ("Support Vector Classifier", SVC(probability=True, random_state=42)),
        'randomforest': ("Random Forest", RandomForestClassifier(random_state=42)),
        'gradientboosting': ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        'histgradientboosting': ("Histogram Gradient Boosting", HistGradientBoostingClassifier(random_state=42)),
        'mlp': ("Multi-Layer Perceptron", MLPClassifier(random_state=42, max_iter=1000)),
        'xgboost': ("XGBoost", XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss')),
        'catboost': ("CatBoost", CatBoostClassifier(random_state=42, verbose=False))
    }
    
    # Configuração do experimento baseado no modo escolhido
    classification_type = "multiclass" if use_multiclass else "binary"
    
    if model_names:
        # Executar modelos específicos
        results = []
        for model_name in model_names:
            if use_default:
                # Executar modelo padrão específico
                model_display_name, model_instance = default_model_map[model_name]
                print(f"\nCONFIGURAÇÃO DO EXPERIMENTO (PARÂMETROS PADRÃO):")
                print(f"  Modelo: {model_display_name}")
                print(f"  Tipo: {classification_type}")
                print(f"  Balanceamento: {balance_strategy}")
                
                experiment_folder = generate_experiment_folder_name(data_source, "default", classification_type)
                experiment_folder = experiment_folder + "_" + balance_strategy if balance_strategy != 'none' else experiment_folder
                experiment_dir = os.path.join("./results", experiment_folder)
                os.makedirs(experiment_dir, exist_ok=True)
                
                try:
                    result = evaluate_model_default(model_instance, model_display_name, X, y, experiment_dir, classification_type, balance_strategy)
                    results.append(result)
                    print(f"✓ {model_display_name} executado com sucesso!")
                except Exception as e:
                    print(f"✗ Erro ao executar {model_display_name}: {e}")
                    results.append({
                        'model_name': model_display_name,
                        'status': 'error',
                        'error': str(e)
                    })
            else:
                # Executar otimização de modelo específico
                model_display_name, optimizer_func = model_map[model_name]
                print(f"\nCONFIGURAÇÃO DO EXPERIMENTO (OTIMIZAÇÃO):")
                print(f"  Modelo: {model_display_name}")
                print(f"  Trials: {N_TRIALS}")
                print(f"  Tipo: {classification_type}")
                print(f"  Parâmetros: {'Reduzido' if use_less_params else 'Completo'}")
                
                result = run_single_model_optimize(model_display_name, optimizer_func, X, y, n_trials=N_TRIALS, data_source=data_source, classification_type=classification_type, use_less_params=use_less_params)
                results.append(result)
        
        # Save summary for default models if applicable
        if use_default:
            from src.reports import save_default_experiment_summary
            save_default_experiment_summary(experiment_dir, results, balance_strategy)
    else:
        # Executar todos os modelos
        if use_default:
            print(f"\nCONFIGURAÇÃO DO EXPERIMENTO (PARÂMETROS PADRÃO):")
            print(f"  Executando: Todos os modelos")
            print(f"  Tipo: {classification_type}")
            print(f"  Balanceamento: {balance_strategy}")
            run_all_default_models(X, y, data_source, classification_type, balance_strategy)
        else:
            print(f"\nCONFIGURAÇÃO DO EXPERIMENTO (OTIMIZAÇÃO):")
            print(f"  Executando: Todos os modelos")
            print(f"  Trials por modelo: {N_TRIALS}")
            print(f"  Tipo: {classification_type}")
            print(f"  Parâmetros: {'Reduzido' if use_less_params else 'Completo'}")
            results = run_all_models_optimize(X, y, n_trials=N_TRIALS, data_source=data_source, classification_type=classification_type, use_less_params=use_less_params)
    
    
    
    print("\nEXPERIMENTO CONCLUÍDO!")


if __name__ == "__main__":
    # Processa argumentos da linha de comando
    args = parse_arguments()
    
    # Executa o experimento com as opções escolhidas
    main(args.renan, args.multiclass, args.default, args.balancedata, args.model, args.lessparams)
