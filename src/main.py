#!/usr/bin/env python3
"""
Arquivo principal para experimentação com modelos de classificação
para predição de genes-alvo usando dados ômicos.
"""

import os
import sys
import argparse
import warnings
from itertools import combinations
sys.path.append('/Users/i583975/git/tcc')

from processing import prepare_dataset, prepare_renan_data
from models import (
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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from evaluation import evaluate_model_default
from reports import generate_experiment_folder_name, set_experiment_timestamp
warnings.filterwarnings('ignore')

# Check GPU availability for XGBoost
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

N_TRIALS = 30


def get_data_paths(use_renan=False):
    if use_renan:
        print("Usando dados do RENAN")
        return ("./renan/data_files/omics_features/UNION_features.tsv",
                "./renan/data_files/labels/UNION_labels.tsv",
                "RENAN")
    else:
        print("Usando dados da ANA")
        return ("./data/UNION_features.tsv",
                "./data/processed/UNION_labels.tsv",
                "ANA")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Experimentação com modelos de classificação')
    parser.add_argument('-renan', '--renan', action='store_true', help='Usa dados do Renan')
    parser.add_argument('-multiclass', '--multiclass', action='store_true', help='Classificação multiclasse')
    parser.add_argument('-default', '--default', action='store_true', help='Modelos padrão')
    parser.add_argument('-balancedata', '--balancedata', type=str,
                        choices=['none', 'smoteenn', 'smotetomek', 'randomundersampler', 'tomeklinks',
                                 'smoten', 'adasyn', 'kmeanssmote'], default='none')
    parser.add_argument('-model', '--model', type=str, nargs='+',
                        choices=['catboost', 'decisiontree', 'gradientboosting', 'histgradientboosting', 'knn',
                                 'mlp', 'randomforest', 'svc', 'xgboost'], default=None)
    parser.add_argument('-lessparams', '--lessparams', action='store_true', help='Conjunto reduzido de parâmetros')
    parser.add_argument('-all-omics-combinations', '--all-omics-combinations', action='store_true', 
                        help='Executa todas as combinações de 2 e 3 ômicas')
    return parser.parse_args()


def run_single_model(model_name, func_or_model, X, y, n_trials=N_TRIALS, is_default=False,
                     classification_type='binary', use_less_params=False, balance_strategy='none', data_source='ANA', omics_to_use=None):
    print("="*80)
    print(f"EXECUTANDO MODELO: {model_name} | Default: {is_default}")
    print(f"Balanceamento: {balance_strategy}")
    print("="*80)

    experiment_folder = generate_experiment_folder_name(data_source, "default" if is_default else "optimized", classification_type)
    if balance_strategy != 'none':
        experiment_folder += f"_{balance_strategy}"
    experiment_dir = os.path.join("./results", experiment_folder)
    os.makedirs(experiment_dir, exist_ok=True)

    try:
        if is_default:
            result = evaluate_model_default(func_or_model, model_name, X, y, experiment_dir, classification_type, balance_strategy, omics_used=omics_to_use)
        else:
            fixed_params = {'balance_strategy': balance_strategy} if balance_strategy != 'none' else None
            result_model, test_metrics = func_or_model(
                X, y, n_trials=n_trials, save_results=True, fixed_params=fixed_params,
                data_source=data_source, classification_type=classification_type,
                use_nested_cv=True, outer_cv_folds=5, use_less_params=use_less_params
            )
            result = {'model_name': model_name, 'status': 'success', 'model': result_model, 'test_metrics': test_metrics}
        
        print(f"[OK] {model_name} executado com sucesso!")
    
    except Exception as e:
        print(f"[ERRO] {model_name}: {e}")
        result = {'model_name': model_name, 'status': 'error', 'error': str(e), 'model': None}

    return result


def main(use_renan=False, use_multiclass=False, use_default=False, balance_strategy='none', model_names=None, use_less_params=False, omics_to_use=None):
    print("CLASSIFICAÇÃO DE GENES-ALVO USANDO DADOS ÔMICOS")
    print("="*80)
    print(f"Timestamp: {set_experiment_timestamp()}")

    features_path, labels_path, data_source = get_data_paths(use_renan)
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Arquivos não encontrados. Abortando.")
        return

    print("Carregando dados...")
    if use_renan:
        X, y, *_ = prepare_renan_data()
    else:
        classification_type = 'multiclass' if use_multiclass else 'binary'

        # Copy Number Alteration, Gene Expression, DNA Methylation, Molecular Function
        # omics_to_use = ['CNA', 'GE', 'METH', 'MF']
        if omics_to_use is None:
            omics_to_use = ['CNA']

        X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path, classification_type, omics_to_use=omics_to_use)

    classification_type = 'multiclass' if use_multiclass else 'binary'

    # Mapas de modelos
    optimize_models = {
        'decisiontree': optimize_decision_tree_classifier,
        'randomforest': optimize_random_forest_classifier,
        'gradientboosting': optimize_gradient_boosting_classifier,
        'histgradientboosting': optimize_hist_gradient_boosting_classifier,
        'knn': optimize_knn_classifier,
        'mlp': optimize_mlp_classifier,
        'svc': optimize_svc_classifier,
        'xgboost': optimize_xgboost_classifier,
        'catboost': optimize_catboost_classifier
    }

    default_models = {
        'decisiontree': DecisionTreeClassifier(random_state=42),
        'randomforest': RandomForestClassifier(random_state=42),
        'gradientboosting': GradientBoostingClassifier(random_state=42),
        'histgradientboosting': HistGradientBoostingClassifier(random_state=42),
        'knn': KNeighborsClassifier(),
        'mlp': MLPClassifier(random_state=42, max_iter=1000),
        'svc': SVC(probability=True, random_state=42),
        'xgboost': XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss', tree_method="gpu_hist" if GPU_AVAILABLE else "hist"),
        'catboost': CatBoostClassifier(random_state=42, verbose=False)
    }

    selected_models = model_names or default_models.keys()

    results = []
    for m in selected_models:
        if use_default:
            results.append(run_single_model(m, default_models[m], X, y, is_default=True,
                                            classification_type=classification_type,
                                            balance_strategy=balance_strategy,
                                            data_source=data_source,
                                            omics_to_use=omics_to_use))
        else:
            results.append(run_single_model(m, optimize_models[m], X, y, n_trials=N_TRIALS,
                                            is_default=False,
                                            classification_type=classification_type,
                                            use_less_params=use_less_params,
                                            balance_strategy=balance_strategy,
                                            data_source=data_source))

    if use_default:
        from reports import save_default_experiment_summary
        experiment_folder = generate_experiment_folder_name(data_source, "default", classification_type)
        if balance_strategy != 'none':
            experiment_folder += f"_{balance_strategy}"
        save_default_experiment_summary(os.path.join("./results", experiment_folder), results, balance_strategy)

    print("\nEXPERIMENTO CONCLUÍDO!")


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.all_omics_combinations:
        # Gera todas as combinações de 2 e 3 ômicas
        all_omics = ['CNA', 'GE', 'METH', 'MF']
        omics_combinations = list(combinations(all_omics, 2)) + list(combinations(all_omics, 3))
        
        print("EXECUTANDO TODAS AS COMBINAÇÕES DE ÔMICAS")
        print("="*80)
        print(f"Total de combinações: {len(omics_combinations)}")
        print(f"Combinações: {omics_combinations}")
        print("="*80 + "\n")
        
        for i, omics_combo in enumerate(omics_combinations, 1):
            print(f"\n{'#'*80}")
            print(f"COMBINAÇÃO {i}/{len(omics_combinations)}: {', '.join(omics_combo)}")
            print(f"{'#'*80}\n")
            
            main(args.renan, args.multiclass, args.default, args.balancedata, args.model, 
                 args.lessparams, omics_to_use=list(omics_combo))
    else:
        main(args.renan, args.multiclass, args.default, args.balancedata, args.model, args.lessparams)
