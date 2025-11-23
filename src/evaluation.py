from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import os
from src.reports import default_report
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model_default(model, model_name, X, y, experiment_dir, classification_type="binary", balance_strategy="none"):
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
    # Return result dict for summary
    return {
        'model_name': model_name,
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'balance_strategy': balance_strategy
    }


# def get_metrics(y_true, y_pred, y_proba, classification_type):
    
#     if classification_type == 'multiclass':
#         # Use per_class_auc_scores for multiclass ROC-AUC and PR-AUC
#         # We need model, X_train, y_train, X_test, y_test, but here we only have y_true, y_pred, y_proba
#         # So we use y_true and y_proba for per-class calculation
#         # Classes (sorted)
#         classes = np.unique(y_true)
#         from sklearn.preprocessing import label_binarize
#         y_true_bin = label_binarize(y_true, classes=classes)
#         per_class_roc_auc = {}
#         per_class_pr_auc = {}
#         for i, label in enumerate(classes):
#             roc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
#             pr = average_precision_score(y_true_bin[:, i], y_proba[:, i])
#             per_class_roc_auc[label] = roc
#             per_class_pr_auc[label] = pr
#         roc_auc = per_class_roc_auc
#         pr_auc = per_class_pr_auc
#         average = 'weighted'
#     else:
#         average = 'binary'
#         roc_auc = roc_auc_score(y_true, y_proba[:,1])
#         pr_auc = average_precision_score(y_true, y_proba[:,1])

#     metrics = {
#         'accuracy': accuracy_score(y_true, y_pred),
#         'precision': precision_score(y_true, y_pred, average=average),
#         'recall': recall_score(y_true, y_pred, average=average),
#         'f1': f1_score(y_true, y_pred, average=average),
#         'roc_auc': roc_auc,
#         'pr_auc': pr_auc,
#         'confusion_matrix': confusion_matrix(y_true, y_pred)
#     }

#     return metrics

# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_auc_score, average_precision_score
# import numpy as np

# def per_class_auc_scores(model, X_train, y_train, X_test, y_test):
#     """
#     Returns:
#         per_class_roc_auc: {class_label: roc_auc}
#         per_class_pr_auc: {class_label: pr_auc}
#     """

#     # Train once (correct)
#     model.fit(X_train, y_train)

#     # Get probabilities
#     y_proba = model.predict_proba(X_test)

#     # Classes (sorted)
#     classes = model.classes_
#     n_classes = len(classes)

#     # Convert true labels to one-vs-rest format
#     y_test_bin = label_binarize(y_test, classes=classes)

#     per_class_roc_auc = {}
#     per_class_pr_auc = {}

#     for i, label in enumerate(classes):
#         # One-vs-rest ROC-AUC
#         roc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
#         pr = average_precision_score(y_test_bin[:, i], y_proba[:, i])

#         per_class_roc_auc[label] = roc
#         per_class_pr_auc[label] = pr

#     return per_class_roc_auc, per_class_pr_auc


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
