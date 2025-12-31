#!/usr/bin/env python3
"""
Script to plot OvR (One-vs-Rest) ROC and PR curves for Gradient Boosting and MLP multiclass models
Combined on the same graphs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Configure matplotlib
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 18

# Colors for each class - GB uses vibrant colors, MLP uses darker/muted versions
GB_COLORS = ['#FF0000', '#FF7F00', '#00CC00']  # Red, Orange, Green
MLP_COLORS = ['#CC0099', '#0066FF', '#FFD700']  # Purple, Blue, Gold
CLASS_NAMES = ['Passenger', 'TSG', 'Oncogenes']


def plot_combined_ovr_roc_curves(metrics_files, model_names, output_dir):
    """Plot combined One-vs-Rest ROC curves for both models"""
    plt.figure(figsize=(14, 10))
    
    n_classes = 3
    line_styles = ['-', '--']  # Solid for GB, dashed for MLP
    color_sets = [GB_COLORS, MLP_COLORS]  # Different colors for each model
    
    for model_idx, (metrics_file, model_name) in enumerate(zip(metrics_files, model_names)):
        # Load metrics
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        y_true = np.array(data['test_predictions']['y_true'])
        y_pred_proba = np.array(data['test_predictions']['y_pred_proba'])
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            color = color_sets[model_idx][i]
            linestyle = line_styles[model_idx]
            label = f'{model_name} - {CLASS_NAMES[i]} (AUC={roc_auc:.3f})'
            
            plt.plot(fpr, tpr, color=color, lw=3, linestyle=linestyle, label=label)
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    plt.legend(loc="lower right", fontsize=22, framealpha=0.9, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'combined_ovr_roc.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_ovr_roc.pdf', bbox_inches='tight')
    plt.close()
    
    print("  Saved combined OvR ROC curve")


def plot_combined_ovr_pr_curves(metrics_files, model_names, output_dir):
    """Plot combined One-vs-Rest PR curves for both models"""
    plt.figure(figsize=(14, 10))
    
    n_classes = 3
    line_styles = ['-', '--']  # Solid for GB, dashed for MLP
    color_sets = [GB_COLORS, MLP_COLORS]  # Different colors for each model
    
    for model_idx, (metrics_file, model_name) in enumerate(zip(metrics_files, model_names)):
        # Load metrics
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        y_true = np.array(data['test_predictions']['y_true'])
        y_pred_proba = np.array(data['test_predictions']['y_pred_proba'])
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        
        # Compute PR curve and PR area for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            pr_auc = auc(recall, precision)
            
            color = color_sets[model_idx][i]
            linestyle = line_styles[model_idx]
            label = f'{model_name} - {CLASS_NAMES[i]} (AUC={pr_auc:.3f})'
            
            plt.plot(recall, precision, color=color, lw=3, linestyle=linestyle, label=label)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.legend(loc="upper right", fontsize=22, framealpha=0.9, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'combined_ovr_pr.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_ovr_pr.pdf', bbox_inches='tight')
    plt.close()
    
    print("  Saved combined OvR PR curve")


def plot_ovr_roc_curves(metrics_file, output_dir, model_name):
    """Plot One-vs-Rest ROC curves for each class"""
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    y_true = np.array(data['test_predictions']['y_true'])
    y_pred_proba = np.array(data['test_predictions']['y_pred_proba'])
    
    n_classes = 3
    
    # Binarize the output
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 9))
    
    for i in range(n_classes):
        color = GB_COLORS[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=3,
                label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    plt.title(f'{model_name} - OvR ROC Curves', fontsize=22)
    plt.legend(loc="lower right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{model_name.lower().replace(" ", "_")}_ovr_roc.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name.lower().replace(" ", "_")}_ovr_roc.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved OvR ROC curve for {model_name}")


def plot_ovr_pr_curves(metrics_file, output_dir, model_name):
    """Plot One-vs-Rest PR curves for each class"""
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    y_true = np.array(data['test_predictions']['y_true'])
    y_pred_proba = np.array(data['test_predictions']['y_pred_proba'])
    
    n_classes = 3
    
    # Binarize the output
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    # Compute PR curve and PR area for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
    
    # Plot
    plt.figure(figsize=(12, 9))
    
    for i in range(n_classes):
        color = MLP_COLORS[i]
        plt.plot(recall[i], precision[i], color=color, lw=3,
                label=f'{CLASS_NAMES[i]} (AUC = {pr_auc[i]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.title(f'{model_name} - OvR PR Curves', fontsize=22)
    plt.legend(loc="upper right", fontsize=22, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{model_name.lower().replace(" ", "_")}_ovr_pr.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_name.lower().replace(" ", "_")}_ovr_pr.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved OvR PR curve for {model_name}")


def main():
    """Main function"""
    base_dir = Path('/Users/i583975/git/tcc/results_final_tcc')
    
    # Find the most recent trained_models directory
    trained_dirs = sorted(base_dir.glob('trained_models_*'))
    if not trained_dirs:
        print("No trained_models directories found!")
        return
    
    latest_dir = trained_dirs[-1]
    multiclass_dir = latest_dir / 'multiclass'
    
    if not multiclass_dir.exists():
        print(f"Multiclass directory not found: {multiclass_dir}")
        return
    
    print("="*80)
    print("GENERATING COMBINED OvR CURVES FOR GRADIENT BOOSTING AND MLP")
    print("="*80)
    print(f"\nReading from: {multiclass_dir}")
    
    # Output directory
    output_dir = multiclass_dir / 'ovr_curves'
    output_dir.mkdir(exist_ok=True)
    print(f"Saving to: {output_dir}\n")
    
    # Models to process
    models = {
        'gradient_boosting': 'Gradient Boosting',
        'mlp': 'MLP'
    }
    
    metrics_files = []
    model_names = []
    
    for model_key, model_name in models.items():
        metrics_file = multiclass_dir / f'metrics_{model_key}.json'
        
        if not metrics_file.exists():
            print(f"⚠️  Metrics file not found: {metrics_file}")
            continue
        
        metrics_files.append(metrics_file)
        model_names.append(model_name)
    
    if len(metrics_files) < 2:
        print("Error: Need both model metrics files!")
        return
    
    print("Processing combined plots...")
    
    # Generate combined OvR ROC curves
    plot_combined_ovr_roc_curves(metrics_files, model_names, output_dir)
    
    # Generate combined OvR PR curves
    plot_combined_ovr_pr_curves(metrics_files, model_names, output_dir)
    
    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nCombined curves saved in: {output_dir}")
    print("  - combined_ovr_roc.png/pdf")
    print("  - combined_ovr_pr.png/pdf")


if __name__ == '__main__':
    main()
