from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import os
from src.reports import default_report, format_5fold_report
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_model_holdout_cv(
    model,
    model_name,
    X, y,
    experiment_dir,
    classification_type="binary",
    balance_strategy="none",
    n_folds=5
):
    """
    80/20 split
    -> run 5-fold CV on the 80%
    -> return metrics for each fold (train + validation)
    -> return aggregated mean validation metrics
    """

    # ================================================
    # STEP 1 — HOLDOUT 80/20
    # ================================================
    X_train_full, X_test_unused, y_train_full, y_test_unused = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    folds_results = []

    # ================================================
    # STEP 2 — 5-FOLD CV ON THE 80%
    # ================================================
    fold_id = 1
    for train_idx, val_idx in kf.split(X_train_full):

        X_train_fold = X_train_full[train_idx]
        y_train_fold = y_train_full[train_idx]

        X_val_fold = X_train_full[val_idx]
        y_val_fold = y_train_full[val_idx]

        # -------------------------------------------
        # OPTIONAL BALANCING — TRAIN ONLY
        # -------------------------------------------
        if balance_strategy != "none":
            from imblearn.combine import SMOTEENN, SMOTETomek
            from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE
            from imblearn.under_sampling import RandomUnderSampler, TomekLinks

            if balance_strategy == "smoteenn":
                balancer = SMOTEENN(random_state=42)
            elif balance_strategy == "smoten":
                balancer = SMOTE(random_state=42)
            elif balance_strategy == "adasyn":
                balancer = ADASYN(random_state=42)
            elif balance_strategy == "kmeanssmote":
                balancer = KMeansSMOTE(random_state=42)
            elif balance_strategy == "smotetomek":
                balancer = SMOTETomek(random_state=42)
            elif balance_strategy == "randomundersampler":
                balancer = RandomUnderSampler(random_state=42)
            elif balance_strategy == "tomeklinks":
                balancer = TomekLinks()
            else:
                raise ValueError(f"Unrecognized balance strategy: {balance_strategy}")

            X_train_fold, y_train_fold = balancer.fit_resample(X_train_fold, y_train_fold)

        # -------------------------------------------
        # PIPELINE
        # -------------------------------------------
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])

        # TRAIN
        pipeline.fit(X_train_fold, y_train_fold)

        # TRAIN metrics
        y_pred_train = pipeline.predict(X_train_fold)
        y_proba_train = pipeline.predict_proba(X_train_fold)
        fold_train_metrics = get_metrics(
            y_train_fold, y_pred_train, y_proba_train, classification_type
        )

        # VALIDATION metrics
        y_pred_val = pipeline.predict(X_val_fold)
        y_proba_val = pipeline.predict_proba(X_val_fold)
        fold_val_metrics = get_metrics(
            y_val_fold, y_pred_val, y_proba_val, classification_type
        )

        folds_results.append({
            "fold": fold_id,
            "train": fold_train_metrics,
            "val": fold_val_metrics
        })

        fold_id += 1

    # ================================================
    # STEP 3 — AGGREGATE VALIDATION METRICS (MEAN)
    # ================================================
    aggregated = {}

    # use keys from first fold
    metric_keys = folds_results[0]["val"].keys()

    for key in metric_keys:
        # skip things like confusion matrices or dicts
        if isinstance(folds_results[0]["val"][key], (int, float, np.floating)):
            aggregated[key] = float(
                np.mean([fold["val"][key] for fold in folds_results])
            )

    # ================================================
    # STEP 4 — SAVE REPORT
    # ================================================
    model_dir = os.path.join(experiment_dir, model_name.lower().replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)

    report_path = os.path.join(model_dir, "5fold_on_80_results.txt")
    with open(report_path, "w") as f:
        f.write(format_5fold_report(model_name, folds_results, aggregated, classification_type))

    return {
        "folds": folds_results,
        "aggregated": aggregated
    }


def evaluate_model_holdout(model, model_name, X, y, experiment_dir, classification_type="binary", balance_strategy="none"):
    """
    Avalia modelo usando holdout 80%/20%
    """

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply balancing only to training set if requested
    if balance_strategy and balance_strategy != "none":
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE
        from imblearn.under_sampling import RandomUnderSampler, TomekLinks
        # Add more strategies as needed
        if balance_strategy == "smoteenn":
            balancer = SMOTEENN(random_state=42)
        elif balance_strategy == "smoten":
            balancer = SMOTE(random_state=42)
        elif balance_strategy == "adasyn":
            balancer = ADASYN(random_state=42)
        elif balance_strategy == "kmeanssmote": ####NEED FIX
            balancer = KMeansSMOTE(random_state=42)
        elif balance_strategy == "smotetomek":
            balancer = SMOTETomek(random_state=42)
        elif balance_strategy == "randomundersampler":
            balancer = RandomUnderSampler(random_state=42)
        elif balance_strategy == "tomeklinks":
            balancer = TomekLinks()
        else:
            raise ValueError(f"Balance strategy '{balance_strategy}' not supported.")
        X_train, y_train = balancer.fit_resample(X_train, y_train)

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    # Train on 80%
    pipeline.fit(X_train, y_train)

    # Train metrics
    y_pred_train = pipeline.predict(X_train)
    y_proba_train = pipeline.predict_proba(X_train)
    train_metrics = get_metrics(y_train, y_pred_train, y_proba_train, classification_type)

    # Test metrics (20%)
    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)
    test_metrics = get_metrics(y_test, y_pred_test, y_proba_test, classification_type)
    # Add raw predictions to test_metrics for plotting compatibility
    test_metrics['y_true'] = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
    test_metrics['y_pred'] = y_pred_test.tolist() if hasattr(y_pred_test, 'tolist') else list(y_pred_test)
    test_metrics['y_pred_proba'] = y_proba_test.tolist() if hasattr(y_proba_test, 'tolist') else list(y_proba_test)

    model_dir = os.path.join(experiment_dir, model_name.lower().replace(' ', '_'))
    os.makedirs(model_dir, exist_ok=True)
    report_path = os.path.join(model_dir, "default_results.txt")
    default_report(
        model_name=model_name,
        folds_metrics={'train_metrics': train_metrics},
        test_metrics=test_metrics,
        output_path=report_path,
        balance_strategy=balance_strategy
    )
    return {
        'model_name': model_name,
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'balance_strategy': balance_strategy
    }

def evaluate_model_default(model, model_name, X, y, experiment_dir, classification_type="binary", balance_strategy="none"):
    evaluate_model_holdout(
        model, model_name, X, y, experiment_dir,
        classification_type=classification_type,
        balance_strategy=balance_strategy
    )
    evaluate_model_holdout_cv(
        model, model_name, X, y, experiment_dir,
        classification_type=classification_type,
        balance_strategy=balance_strategy
    )

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize


def get_metrics(y_true, y_pred, y_proba, classification_type):
    """
    Returns:
        - global metrics (accuracy, macro/weighted AUCs)
        - per-class ROC-AUC
        - per-class PR-AUC
        - micro/macro/weighted aggregations
    """

    metrics = {}

    # =============================
    # Common metrics
    # =============================
    if classification_type == "multiclass":
        average = "weighted"
    else:
        average = "binary"

    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average)
    metrics["recall"]    = recall_score(y_true, y_pred, average=average)
    metrics["f1"]        = f1_score(y_true, y_pred, average=average)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # =====================================================
    # ===============   BINARY CLASS   ====================
    # =====================================================
    if classification_type == "binary":
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        metrics["pr_auc"]  = average_precision_score(y_true, y_proba[:, 1])

        return metrics  # no per-class needed for binary


    # =====================================================
    # =============== MULTICLASS CLASS ====================
    # =====================================================

    classes = np.unique(y_true)
    n_classes = len(classes)

    # One-hot encode y_true
    y_true_bin = label_binarize(y_true, classes=classes)

    # -----------------------------------------------------
    # Per-class AUCs
    # -----------------------------------------------------
    per_class_roc = {}
    per_class_pr  = {}

    for idx, cls in enumerate(classes):

        # ROC-AUC per class
        per_class_roc[cls] = roc_auc_score(
            y_true_bin[:, idx],
            y_proba[:, idx]
        )

        # PR-AUC per class
        per_class_pr[cls] = average_precision_score(
            y_true_bin[:, idx],
            y_proba[:, idx]
        )

    metrics["per_class_roc_auc"] = per_class_roc
    metrics["per_class_pr_auc"]  = per_class_pr

    # -----------------------------------------------------
    # Aggregated ROC-AUC (official: macro / weighted)
    # -----------------------------------------------------
    class_counts = np.bincount(y_true)
    weights = class_counts / class_counts.sum()

    metrics["roc_auc_macro"]   = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    metrics["roc_auc_weighted"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    metrics["roc_auc_micro"]   = roc_auc_score(y_true, y_proba, multi_class="ovr", average="micro")

    # -----------------------------------------------------
    # Aggregated PR-AUC
    # -----------------------------------------------------
    # (1) Macro-average PR-AUC
    metrics["pr_auc_macro"] = np.mean(list(per_class_pr.values()))

    # (2) Weighted PR-AUC
    weighted_pr = 0
    for w, cls in zip(weights, classes):
        weighted_pr += w * per_class_pr[cls]
    metrics["pr_auc_weighted"] = weighted_pr

    # (3) Micro-average PR-AUC — THEORETICALLY CORRECT
    metrics["pr_auc_micro"] = average_precision_score(y_true_bin, y_proba)

    # ALSO: a recommended single metric for tables
    metrics["pr_auc"] = metrics["pr_auc_micro"]
    metrics["roc_auc"] = metrics["roc_auc_weighted"]

    return metrics
