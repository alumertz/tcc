import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from reports import default_report, format_5fold_report

def get_balancer(strategy: str):
    """Retorna o balanceador correspondente à estratégia escolhida."""
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks

    balancers = {
        "smoteenn": SMOTEENN(random_state=42),
        "smoten": SMOTE(random_state=42),
        "adasyn": ADASYN(random_state=42),
        "kmeanssmote": KMeansSMOTE(random_state=42),
        "smotetomek": SMOTETomek(random_state=42),
        "randomundersampler": RandomUnderSampler(random_state=42),
        "tomeklinks": TomekLinks()
    }

    if strategy not in balancers:
        raise ValueError(f"Unrecognized balance strategy: {strategy}")
    
    return balancers[strategy]

def build_pipeline(model):
    """Cria pipeline com scaler e modelo."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

def get_metrics(y_true, y_pred, y_proba=None, classification_type="binary"):
    """Calcula métricas principais para binário e multi-classe."""
    average = "macro" if classification_type == "multiclass" else "binary"
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    if y_proba is None:
        return metrics

    y_true_arr = np.array(y_true)
    y_proba_arr = np.array(y_proba)

    if classification_type == "binary":
        metrics["roc_auc"] = roc_auc_score(y_true_arr, y_proba_arr[:, 1])
        metrics["pr_auc"] = average_precision_score(y_true_arr, y_proba_arr[:, 1])
    else:
        classes = np.unique(y_true_arr)
        y_true_bin = label_binarize(y_true_arr, classes=classes)
        per_class_roc = {cls: roc_auc_score(y_true_bin[:, i], y_proba_arr[:, i]) 
                         for i, cls in enumerate(classes)}
        per_class_pr  = {cls: average_precision_score(y_true_bin[:, i], y_proba_arr[:, i])
                         for i, cls in enumerate(classes)}
        metrics.update({
            "per_class_roc_auc": per_class_roc,
            "per_class_pr_auc": per_class_pr,
            "roc_auc_macro": roc_auc_score(y_true_arr, y_proba_arr, multi_class="ovr", average="macro"),
            "roc_auc_weighted": roc_auc_score(y_true_arr, y_proba_arr, multi_class="ovr", average="weighted"),
            "roc_auc_micro": roc_auc_score(y_true_arr, y_proba_arr, multi_class="ovr", average="micro"),
            "pr_auc_macro": np.mean(list(per_class_pr.values())),
            "pr_auc_weighted": np.average(list(per_class_pr.values()), weights=np.bincount(y_true_arr)/len(y_true_arr)),
            "pr_auc_micro": average_precision_score(y_true_bin, y_proba_arr),
        })
    return metrics

def evaluate_model_holdout(model, model_name, X, y, experiment_dir, classification_type="binary", 
                           balance_strategy="none", omics_used=None):
    """Treina e avalia modelo usando holdout (80/20)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if balance_strategy != "none":
        balancer = get_balancer(balance_strategy)
        X_train, y_train = balancer.fit_resample(X_train, y_train)

    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    # Métricas
    train_metrics = get_metrics(y_train, pipeline.predict(X_train), pipeline.predict_proba(X_train), classification_type)
    test_metrics = get_metrics(y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test), classification_type)

    # Adiciona outputs
    test_metrics.update({
        'y_true': y_test.tolist(),
        'y_pred': pipeline.predict(X_test).tolist(),
        'y_pred_proba': pipeline.predict_proba(X_test).tolist()
    })

    # Salvar relatório
    model_dir = os.path.join(experiment_dir, model_name.lower().replace(' ', '_'))
    os.makedirs(model_dir, exist_ok=True)
    default_report(
        model_name=model_name,
        folds_metrics={'train_metrics': train_metrics},
        test_metrics=test_metrics,
        output_path=os.path.join(model_dir, "default_results.txt"),
        balance_strategy=balance_strategy,
        omics_used=omics_used
    )

    return {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'balance_strategy': balance_strategy
    }

def evaluate_model_holdout_cv(  model, model_name, X, y, experiment_dir, classification_type="binary", 
                                balance_strategy="none", n_folds=5, omics_used=None):
    
    """Treina e avalia modelo usando KFold CV dentro do holdout."""
    
    X_train_full, _, y_train_full, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds_results = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
        X_train_fold, y_train_fold = X_train_full[train_idx], y_train_full[train_idx]
        X_val_fold, y_val_fold = X_train_full[val_idx], y_train_full[val_idx]

        if balance_strategy != "none":
            balancer = get_balancer(balance_strategy)
            X_train_fold, y_train_fold = balancer.fit_resample(X_train_fold, y_train_fold)

        pipeline = build_pipeline(model)
        pipeline.fit(X_train_fold, y_train_fold)

        folds_results.append({
            "fold": fold_id,
            "train": get_metrics(y_train_fold, pipeline.predict(X_train_fold), pipeline.predict_proba(X_train_fold), classification_type),
            "val": get_metrics(y_val_fold, pipeline.predict(X_val_fold), pipeline.predict_proba(X_val_fold), classification_type)
        })

    # Métricas agregadas
    aggregated = {}
    metric_keys = folds_results[0]["val"].keys()
    for key in metric_keys:
        if isinstance(folds_results[0]["val"][key], (int, float, np.floating)):
            aggregated[key] = float(np.mean([fold["val"][key] for fold in folds_results]))

    # Salvar relatório
    model_dir = os.path.join(experiment_dir, model_name.lower().replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "5fold_on_80_results.txt"), "w") as f:
        f.write(format_5fold_report(model_name, folds_results, aggregated, classification_type))

    return {"folds": folds_results, "aggregated": aggregated}

def evaluate_model_default( model, model_name, X, y, experiment_dir, classification_type="binary", 
                            balance_strategy="none", omics_used=None):
    
    """Avalia modelo usando holdout + CV."""

    return {
        "holdout": evaluate_model_holdout(model, model_name, X, y, experiment_dir,
                                          classification_type, balance_strategy, omics_used=omics_used),
        "cv": evaluate_model_holdout_cv(model, model_name, X, y, experiment_dir,
                                        classification_type, balance_strategy, omics_used=omics_used)
    }
