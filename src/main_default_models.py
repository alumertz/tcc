#!/usr/bin/env python3
"""
Avaliação de modelos com parâmetros padrão para comparação com modelos otimizados.
"""

import os
import json
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from processing import prepare_dataset, get_dataset_info

warnings.filterwarnings('ignore')

RESULTS_DIR = "results/omics"
SCALING_REQUIRED = {"mlp", "knn"}
CV_SPLITS = 5
RANDOM_STATE = 42


def build_pipeline(model_name: str, model: BaseEstimator) -> Pipeline:
    """Cria um pipeline com ou sem escalonamento."""
    if model_name.lower() in SCALING_REQUIRED:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])
    return Pipeline([("classifier", model)])


def perform_cross_validation(pipeline: Pipeline, X, y) -> Dict[str, Any]:
    """Executa validação cruzada e retorna métricas com média e std."""
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision'
    }

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results_raw = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

    return {
        metric: {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores.tolist()
        }
        for metric, scores in cv_results_raw.items() if metric.startswith('test_')
    }


def evaluate_model_default(model: BaseEstimator, model_name: str, X, y, save_results: bool = True) -> Dict:
    """Avalia um modelo com parâmetros padrão usando holdout e cross-validation."""
    print(f"\n{'='*80}\nAVALIANDO MODELO: {model_name.upper()}\n{'='*80}")
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Conjunto de treino+validação: {X_trainval.shape[0]} amostras")
    print(f"Conjunto de teste: {X_test.shape[0]} amostras")

    pipeline = build_pipeline(model_name, model)

    print("\nExecutando validação cruzada 5-fold...")
    cv_results = perform_cross_validation(pipeline, X_trainval, y_trainval)

    for metric, result in cv_results.items():
        print(f"  {metric.upper()}: {result['mean']:.4f} ± {result['std']:.4f}")

    print("\nTreinando no conjunto completo e avaliando no teste...")
    pipeline.fit(X_trainval, y_trainval)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }

    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    class_report = classification_report(y_test, y_pred, target_names=['Non-driver', 'Driver'])
    print(f"\nRelatório de classificação:\n{class_report}")

    if save_results:
        save_model_results(model_name, cv_results, test_metrics, class_report, pipeline)

    return {
        'model_name': model_name,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'classification_report': class_report,
        'status': 'success',
        'model': pipeline
    }


def save_model_results(model_name: str, cv_results: Dict, test_metrics: Dict, class_report: str, pipeline: Pipeline):
    """Salva os resultados da avaliação em TXT e JSON."""
    model_dir = os.path.join(RESULTS_DIR, model_name.lower().replace(' ', '_'))
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = {
        k.replace('classifier__', ''): v for k, v in pipeline.get_params().items()
        if k.startswith('classifier__')
    }

    # Salva TXT
    with open(os.path.join(model_dir, f"default_results_{timestamp}.txt"), 'w') as f:
        f.write(f"RESULTADOS DO MODELO: {model_name.upper()} (PARÂMETROS PADRÃO)\n")
        f.write("="*80 + "\n")
        f.write(f"Data/Hora: {datetime.now()}\n\n")
        f.write("PARÂMETROS UTILIZADOS:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nVALIDAÇÃO CRUZADA (5-FOLD):\n")
        for metric, res in cv_results.items():
            f.write(f"{metric.upper()}: {res['mean']:.4f} ± {res['std']:.4f}\n")
        f.write("\nMÉTRICAS NO TESTE:\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nRELATÓRIO DE CLASSIFICAÇÃO:\n" + class_report)

    # Salva JSON
    with open(os.path.join(model_dir, f"default_metrics_{timestamp}.json"), 'w') as f:
        json.dump({
            'model_name': model_name,
            'timestamp': timestamp,
            'parameters': params,
            'cv_results': cv_results,
            'test_metrics': test_metrics
        }, f, indent=2)


def run_all_default_models(X, y) -> List[Dict]:
    """Executa todos os modelos e retorna os resultados."""
    models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=30)),
        ("Random Forest", RandomForestClassifier(random_state=30)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=30)),
        ("Histogram Gradient Boosting", HistGradientBoostingClassifier(random_state=30)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Multi-Layer Perceptron", MLPClassifier(random_state=30, max_iter=1000)),
    ]

    print("INICIANDO AVALIAÇÃO COM PARÂMETROS PADRÃO\n")
    results = []

    for i, (name, model) in enumerate(models, 1):
        print(f"({i}/{len(models)}) Executando: {name}")
        try:
            result = evaluate_model_default(model, name, X, y)
        except Exception as e:
            print(f"❌ Erro ao executar {name}: {e}")
            result = {'model_name': name, 'status': 'error', 'error': str(e)}
        results.append(result)
        time.sleep(1)
    return results


def summarize_default_results(results: List[Dict]):
    """Gera resumo final dos resultados."""
    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS")
    print("="*80)

    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']

    print(f"✅ Sucesso: {len(success)}")
    print(f"❌ Falhas: {len(failed)}\n")

    if success:
        print


def main():
    """Função principal para rodar avaliação com modelos padrão."""
    print("="*80)
    print("AVALIAÇÃO DE GENES-ALVO COM PARÂMETROS PADRÃO")
    print("="*80)

    features_path = "renan/data_files/omics_features/UNION_features.tsv"
    labels_path = "renan/data_files/labels/UNION_labels.tsv"

    # Verificação de arquivos
    if not os.path.exists(features_path):
        print(f"❌ Arquivo de features não encontrado: {features_path}")
        return

    if not os.path.exists(labels_path):
        print(f"❌ Arquivo de labels não encontrado: {labels_path}")
        return

    print("✅ Carregando e preparando dados...")
    X, y, gene_names, feature_names = prepare_dataset(features_path, labels_path)

    if X is None:
        print("❌ Erro ao preparar dataset. Abortando.")
        return

    # Mostrar informações básicas
    info = get_dataset_info(X, y, gene_names, feature_names)
    print("\nINFORMAÇÕES DO DATASET:")
    print(f"  Amostras: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Distribuição das classes: {info['class_distribution']}")
    print(f"  Média das features: {info['feature_stats']['mean']:.4f}")
    print(f"  Desvio padrão: {info['feature_stats']['std']:.4f}")
    print(f"  Porcentagem de zeros: {info['feature_stats']['zeros_percentage']:.2f}%")

    print("\nCONFIGURAÇÃO DO EXPERIMENTO:")
    print(f"  Parâmetros: Padrão")
    print(f"  Validação: Holdout 80/20 + Cross-validation estratificada (5 folds)")
    print(f"  Métricas: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC")
    print(f"  Resultados em: {RESULTS_DIR}/<modelo>/default_*.txt/json")

    # Executar modelos
    print("\n🚀 Iniciando execução dos modelos...\n")
    results = run_all_default_models(X, y)

    # Resumo final
    summarize_default_results(results)
    print("\n✅ EXPERIMENTO FINALIZADO COM SUCESSO!")

if __name__ == "__main__":
    main()
