# Gene Classification for Cancer Research

A machine learning pipeline for classifying genes into cancer-related categories using omics features. This project implements binary and multiclass classification to identify oncogenes, tumor suppressor genes (TSGs), and passenger genes.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Processing Flow](#data-processing-flow)
- [Setup and Installation](#setup-and-installation)
- [Available Commands](#available-commands)
- [Data Sources](#data-sources)
- [Machine Learning Models](#machine-learning-models)
- [Output Structure](#output-structure)
- [Visualization and Analysis](#visualization-and-analysis)

## Overview

This project aims to classify genes based on their role in cancer using machine learning approaches. The pipeline supports:

- **Binary Classification**: Cancer genes (1) vs Passenger genes (0), with candidates as NaN
- **Multiclass Classification**: TSG (1) vs Oncogenes (2) vs Passenger genes (0), with candidates as NaN
- **Multiple Data Sources**: Support for both processed (Ana) and original (Renan) data formats
- **8 Machine Learning Models**: From simple decision trees to advanced ensemble methods
- **Two Operation Modes**: Default parameters for quick testing or Optuna optimization for best performance

## Project Structure

```
tcc/
├── src/                           # Source code
│   ├── main.py                    # Main execution script
│   ├── process_data.py            # Data preprocessing pipeline
│   ├── processing.py              # Dataset preparation utilities
│   ├── models.py                  # ML model implementations with optimization
│   ├── evaluation.py              # Model evaluation functions
│   ├── plot_curves.py             # ROC/PR curve visualization
│   ├── plot_curves_multiclass.py  # Multiclass visualization
│   └── reports.py                 # Results summarization
├── data/                          # Main data directory (Ana's format)
│   ├── UNION_features.tsv         # Gene features matrix
│   ├── processed/                 # Processed data files
│   │   ├── UNION_labels.tsv       # Gene labels (2class, 3class)
│   │   ├── canonical_genes.tsv    # Known cancer genes
│   │   └── candidate_genes.tsv    # Candidate genes
│   └── [raw data files]           # OncoKB, NCG, COSMIC, OMIM, etc.
├── renan/                         # Original data format (Renan's)
│   └── data_files/
│       ├── omics_features/        # Original features
│       └── labels/                # Original labels
├── results/                       # Experiment results
│   └── [timestamp_experiment]/    # Timestamped experiment folders
└── README.md                      # This file
```

## Data Processing Flow

### 1. Raw Data Collection (`process_data.py`)
```
Raw Sources → HGNC Validation → Gene Classification → Feature Processing
     ↓              ↓                    ↓                  ↓
- OncoKB        Symbol       Canonical/Candidate    UNION_features.tsv
- NCG           Mapping       Classification        UNION_labels.tsv  
- COSMIC        ↓                    ↓                  ↓
- OMIM          Approved      canonical_genes.tsv   processed/
- HGNC          Symbols       candidate_genes.tsv
```

### 2. Dataset Preparation (`processing.py`)
```
Features + Labels → Alignment → Train/Test Split → Standardization → ML Models
     ↓                ↓              ↓                  ↓            ↓
UNION_features.tsv   Common      80% Train          StandardScaler   8 Models
UNION_labels.tsv     Genes       20% Test           Pipeline         Results
```

### 3. Model Training and Evaluation (`main.py`)
```
Dataset → Cross-Validation → Hyperparameter → Final Model → Results
   ↓           ↓              Optimization       ↓          ↓
Feature    5-Fold CV        Optuna (optional)   Test Set   Metrics
Matrix     Stratified       30 trials/model     Evaluation JSON
```

## Setup and Installation

### Prerequisites
- Python 3.7+
- Virtual environment support

### Installation Steps

1. **Create and activate virtual environment**:
```bash
python3 -m venv mlenv
source mlenv/bin/activate
```

2. **Install required packages**:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy optuna catboost
```

3. **Verify data files exist**:
```bash
ls data/UNION_features.tsv
ls data/processed/UNION_labels.tsv
```

## Available Commands

### Main Execution (`src/main.py`)

#### Basic Usage
```bash
# Binary classification with optimization (default)
python3 src/main.py

# Multiclass classification with optimization  
python3 src/main.py -multiclass

# Quick testing with default parameters
python3 src/main.py -default

# Use original data format (Renan's)
python3 src/main.py -renan

# Combine options
python3 src/main.py -multiclass -default
python3 src/main.py -renan -default
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-multiclass` | Use multiclass classification (TSG vs Oncogene vs Passenger) | Binary (Cancer vs Passenger) |
| `-renan` | Use original data format from renan/ directory | Use processed data from data/ |
| `-default` | Use default model parameters (fast) | Use Optuna optimization (slow) |
| `-h, --help` | Show help message and examples | - |

#### Expected Runtime
- **Default mode**: ~2-5 minutes (8 models with default parameters)
- **Optimization mode**: ~30 minutes per model (~4 hours total for 8 models)

### Data Processing (`src/process_data.py`)

```bash
# Reprocess all data from raw sources
python3 src/process_data.py
```

**What it does**:
- Loads raw data from OncoKB, NCG, COSMIC, OMIM
- Validates gene symbols using HGNC
- Creates canonical_genes.tsv and candidate_genes.tsv
- Generates UNION_labels.tsv with binary (2class) and multiclass (3class) labels
- Processes UNION_features.tsv with approved gene symbols

### Visualization (`src/plot_curves.py`)

```bash
# Generate ROC and PR curves from saved results
python3 src/plot_curves.py
```

**Features**:
- Interactive experiment selection
- Automatic detection of binary vs multiclass
- High-quality PDF and PNG outputs
- Standardized color schemes

## Data Sources

### Gene Labels Sources
- **OncoKB**: Precision oncology knowledge base
- **NCG**: Network of Cancer Genes  
- **COSMIC**: Catalogue of Somatic Mutations in Cancer (CGC)
- **OMIM**: Online Mendelian Inheritance in Man
- **HGNC**: Human Gene Nomenclature Committee (for symbol validation)

### Feature Sources
- **UNION_features.tsv**: Integrated omics features including:
  - Gene expression data
  - Protein-protein interaction networks
  - Pathway information
  - Sequence-based features

### Classification Schemes

#### Binary Classification (2class)
- **Class 1**: Cancer genes (known oncogenes or TSGs)
- **Class 0**: Passenger genes (not associated with cancer but appear in datasets)
- **NaN**: Candidate genes (unknown, used for prediction)

#### Multiclass Classification (3class)  
- **Class 1**: Tumor Suppressor Genes (TSGs)
- **Class 2**: Oncogenes (ONC)
- **Class 0**: Passenger genes
- **NaN**: Candidate genes

## Machine Learning Models

### Available Models (8 total)

1. **Decision Tree** - Simple interpretable model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Sequential boosting algorithm  
4. **Histogram Gradient Boosting** - Optimized gradient boosting
5. **K-Nearest Neighbors** - Instance-based learning
6. **Multi-Layer Perceptron** - Neural network
7. **Support Vector Classifier** - SVM with RBF kernel
8. **CatBoost** - Gradient boosting optimized for categorical features

### Optimization Details

#### Default Mode (`-default`)
- Uses scikit-learn default parameters
- Fast execution (~2-5 minutes total)
- Good for quick testing and prototyping
- Consistent results across runs

#### Optimization Mode (default)
- Uses Optuna for hyperparameter optimization
- 30 trials per model with Bayesian optimization
- Optimizes for PR AUC (Average Precision)
- Nested cross-validation (5-fold outer, 3-fold inner)
- Slow but produces best results (~4 hours total)

### Evaluation Metrics

- **Precision**: Proportion of predicted positives that are correct
- **Recall**: Proportion of actual positives correctly identified  
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve
- **PR AUC**: Area under the Precision-Recall curve (primary optimization metric)

## Output Structure

### Results Directory
```
results/YYYYMMDD_HHMMSS_[source]_[mode]_[classification]/
├── [model_name]/
│   ├── metrics.json           # Detailed metrics and predictions
│   └── results.txt            # Human-readable summary
├── curves/                    # Generated visualizations
│   ├── roc_comparison_*.png   # ROC curves comparison
│   ├── pr_comparison_*.png    # PR curves comparison
│   └── [multiclass curves]/   # Additional curves for multiclass
└── summary_*.txt              # Overall experiment summary
```

### Key Output Files

#### `metrics.json` (per model)
```json
{
  "model_name": "random_forest",
  "best_params": {...},
  "cv_scores": {...},
  "test_metrics": {...},
  "test_predictions": {
    "y_true": [...],
    "y_pred": [...], 
    "y_pred_proba": [...]
  }
}
```

#### `summary_*.txt` (experiment-wide)
- Comparison table of all models
- Best performing model identification
- Runtime statistics
- Configuration details

## Visualization and Analysis

### Automatic Plots Generation

The pipeline automatically generates:

#### Binary Classification
- **ROC Curves**: True Positive Rate vs False Positive Rate
- **PR Curves**: Precision vs Recall (better for imbalanced datasets)

#### Multiclass Classification
- **ROC Curves (OvR)**: One-vs-Rest for each class
- **ROC Curves (OvO)**: One-vs-One for each class pair
- **PR Curves**: For each class
- **Weighted AUC Comparison**: Summary metrics across classes

### Manual Plot Generation

```bash
# Generate plots from existing results
python3 src/plot_curves.py

# Interactive selection of experiment to visualize
# Supports both binary and multiclass automatically
```

## Usage Examples

### Example 1: Quick Testing
```bash
# Fast run with default parameters for binary classification
python3 src/main.py -default

# Expected output: Results in ~3 minutes
# Generated: results/[timestamp]_ana_default_binary/
```

### Example 2: Full Optimization  
```bash
# Full optimization for multiclass classification
python3 src/main.py -multiclass

# Expected output: Results in ~4 hours  
# Generated: results/[timestamp]_ana_optimized_multiclass/
```

### Example 3: Original Data Format
```bash
# Use Renan's original data format
python3 src/main.py -renan -default

# Expected output: Results using renan/data_files/
# Generated: results/[timestamp]_renan_default_binary/
```

### Example 4: Complete Pipeline
```bash
# 1. Reprocess data if needed
python3 src/process_data.py

# 2. Run classification
python3 src/main.py -multiclass

# 3. Generate additional plots
python3 src/plot_curves.py

# Expected output: Complete analysis with all visualizations
```

## Troubleshooting

### Common Issues

1. **Missing data files**:
   ```bash
   # Run data processing first
   python3 src/process_data.py
   ```

2. **Environment issues**:
   ```bash
   # Recreate virtual environment
   rm -rf mlenv
   python3 -m venv mlenv
   source mlenv/bin/activate
   pip install pandas scikit-learn matplotlib seaborn numpy optuna catboost
   ```

3. **Memory issues with large datasets**:
   - Use `-default` mode for faster execution
   - Reduce the number of Optuna trials in `main.py`

4. **Plot generation fails**:
   ```bash
   # Check if results exist
   ls results/
   
   # Run models first
   python3 src/main.py -default
   ```

### Performance Tips

- Use `-default` for development and testing
- Use optimization mode only for final results
- Monitor system resources during optimization
- Results are automatically saved; safe to interrupt and resume

## Contributing

This is a research project. For questions or suggestions:
1. Check existing results in `results/` directory
2. Review the code documentation in source files
3. Test changes with `-default` mode first

## License

Academic research project - see institution guidelines for usage restrictions.
