#!/usr/bin/env python3
"""
Avaliação de modelos com parâmetros padrão para comparação com modelos otimizados via Optuna.
"""

import sys, os, json, warnings
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from processing import prepare_dataset
from reports import generate_enhanced_classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sys.path.append('/Users/i583975/git/tcc')


METRICS = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision'
}


def evaluate_model_default(model, model_name, X, y, save_results=True):
    """Avalia um modelo com parâmetros padrão usando holdout e 5-fold CV."""
    
    print(f"\n{'='*80}\nAVALIANDO MODELO: {model_name.upper()}\n{'='*80}")
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pipeline: escalonamento apenas para MLP e KNN
    if model_name.lower() in ['mlp', 'k-nearest neighbors']:
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])
    else:
        pipeline = Pipeline([("classifier", model)])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
    
    # Validação cruzada
    cv_results = {}
    for metric_name, scoring in METRICS.items():
        scores = cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring=scoring)
        cv_results[metric_name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores.tolist()}
    
    print("Resultados da validação cruzada:")
    for metric, result in cv_results.items():
        print(f"  {metric.upper()}: {result['mean']:.4f} ± {result['std']:.4f}")
    
    # Treino final e avaliação no teste
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
    
    print("Resultados no conjunto de teste:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    class_report = generate_enhanced_classification_report(y_test, y_pred, y_pred_proba)
    
    if save_results:
        save_default_results(model_name, cv_results, test_metrics, class_report, pipeline.get_params())
    
    return {'model_name': model_name, 'cv_results': cv_results, 'test_metrics': test_metrics, 
            'classification_report': class_report, 'model': pipeline}


def save_default_results(model_name, cv_results, test_metrics, class_report, params):
    base_dir = "results"
    model_dir = os.path.join(base_dir, model_name.lower().replace(' ', '_').replace('-', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    classifier_params = {k.replace('classifier__', ''): v for k, v in params.items() if k.startswith('classifier__')}
    
    # Salvar TXT
    results_file = os.path.join(model_dir, f"default_results_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write(f"RESULTADOS DO MODELO: {model_name.upper()} (PADRÃO)\n{'='*80}\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("PARÂMETROS UTILIZADOS:\n" + "-"*50 + "\n")
        for p, v in classifier_params.items(): f.write(f"{p}: {v}\n")
        f.write("\nVALIDAÇÃO CRUZADA:\n" + "-"*50 + "\n")
        for metric, res in cv_results.items():
            f.write(f"{metric.upper()}: Média={res['mean']:.4f}, Std={res['std']:.4f}, Scores={res['scores']}\n")
        f.write("\nTESTE FINAL:\n" + "-"*50 + "\n")
        for metric, val in test_metrics.items():
            f.write(f"{metric.upper()}: {val:.4f}\n")
        f.write("\nRELATÓRIO DETALHADO:\n" + "-"*30 + "\n" + class_report + "\n")
    
    # Salvar JSON
    json_file = os.path.join(model_dir, f"default_metrics_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump({'model_name': model_name, 'timestamp': timestamp,
                   'parameters': classifier_params, 'cv_results': cv_results,
                   'test_metrics': test_metrics}, f, indent=2)
    
    print(f"Resultados salvos em: {results_file} e {json_file}")


def run_all_default_models(X, y):
    models_config = [
        ("Decision Tree", DecisionTreeClassifier(random_state=30)),
        ("Random Forest", RandomForestClassifier(random_state=30)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=30)),
        ("Histogram Gradient Boosting", HistGradientBoostingClassifier(random_state=30)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Multi-Layer Perceptron", MLPClassifier(random_state=30, max_iter=1000)),
        ("Support Vector Classifier", SVC(random_state=30, probability=True)),
        ("CatBoost", CatBoostClassifier(random_state=30, verbose=False, allow_writing_files=False)),
    ]
    
    results = []
    for model_name, model in models_config:
        try:
            results.append(evaluate_model_default(model, model_name))
            print(f"✅ {model_name} executado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao executar {model_name}: {e}")
            results.append({'model_name': model_name, 'status': 'error', 'error': str(e)})
    
    return results


def main():
    features_path = "./data/UNION_features.tsv"
    labels_path = "./data/processed/UNION_labels.tsv"
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Arquivos de dados não encontrados. Abortando.")
        return
    
    X, y, gene_names = prepare_dataset(features_path, labels_path)
    if X is None: return
    
    print(f"\nIniciando avaliação com parâmetros padrão para {X.shape[0]} amostras e {X.shape[1]} features...")
    run_all_default_models(X, y)
    print("\nExperimento concluído. Resultados salvos em 'results/'.")


if __name__ == "__main__":
    main()
