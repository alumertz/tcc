from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import os
from src.reports import default_report
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model_default(model, model_name, X, y, experiment_dir, classification_type="binary"):
    """
    Avalia modelo usando holdout 80%/20%
    """

    # print("y contains NaNs:", np.isnan(y).any())
    # print("y unique values:", np.unique(y))
    # print("X shape:", X.shape, "y shape:", y.shape)
    # print("First 10 values of y:", y[:10])
    # print("Random 10 values of y:", y[np.random.choice(len(y), 10, replace=False)])
    #print("First 10 values of x:", X[:10])
    #print("Random 10 values of x:", X[np.random.choice(len(X), 10, replace=False)])

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # print("Shape de X_train:", X_train.shape)
    # print("Shape de X_test:", X_test.shape)
    # print("Shape de y_train:", y_train.shape)
    # print("Shape de y_test:", y_test.shape)

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
        output_path=report_path
    )


def get_metrics(y_true, y_pred, y_proba, classification_type):

    if classification_type == 'multiclass':
        average = 'weighted'  
    else: average = 'binary'

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average),
        'roc_auc': roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            if classification_type == 'multiclass'
            else roc_auc_score(y_true, y_proba[:, 1]),
        'pr_auc': average_precision_score(y_true, y_proba, average='weighted')
            if classification_type == 'multiclass'
            else average_precision_score(y_true, y_proba[:, 1]),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics
