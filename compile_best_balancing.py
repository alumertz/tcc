import os
import re
from collections import defaultdict
import operator
import sys

# --- Configuration ---
ROOT_DIR = 'results'
RESULTS_FILENAME = '5fold_on_80_results.txt'
OUTPUT_FILENAME = 'aggregated_results.txt'

# --- Metric Definitions ---

# Metrics used for Binary Classification (extracted from the provided example)
BINARY_METRICS = {
    'Acur.': 'accuracy', 
    'Prec.': 'precision', 
    'Recall': 'recall', 
    'F1': 'f1', 
    'AUC-ROC': 'roc_auc', 
    'AUC-PR': 'pr_auc'
}
BINARY_RANK_KEY = 'AUC-PR' # Key used for ranking

# Metrics used for Multiclass Classification (extracted from the provided example)
MULTICLASS_METRICS = {
    'Acur.': 'accuracy', 
    'Prec.': 'precision', 
    'Recall': 'recall', 
    'F1': 'f1', 
    'AUC-ROC': 'roc_auc_macro', # Using macro for main ROC column
    'AUC-PR': 'pr_auc_macro' # Using macro for main PR column
}
MULTICLASS_RANK_KEY = 'AUC-PR' # Key used for ranking

# Regular expression patterns for metric blocks
# Capture group for value name is important here
FLOAT_CAPTURE = r'\s+(\d\.\d+)'

# Regex for Binary Metrics
BINARY_RE = re.compile(
    r'AGGREGATED VALIDATION METRICS \(MEAN OF 5 FOLDS\)\s+'
    r'={60,}\s+'
    r'accuracy:' + FLOAT_CAPTURE + r'\s+'     # Group 1: accuracy
    r'precision:' + FLOAT_CAPTURE + r'\s+'    # Group 2: precision
    r'recall:' + FLOAT_CAPTURE + r'\s+'       # Group 3: recall
    r'f1:' + FLOAT_CAPTURE + r'\s+'           # Group 4: f1
    r'.*?' 
    r'roc_auc:' + FLOAT_CAPTURE + r'\s+'      # Group 5: roc_auc
    r'pr_auc:' + FLOAT_CAPTURE,               # Group 6: pr_auc
    re.MULTILINE | re.DOTALL
)

# Regex for Multiclass Metrics
MULTICLASS_RE = re.compile(
    r'AGGREGATED VALIDATION METRICS \(MEAN OF 5 FOLDS\)\s+'
    r'={60,}\s+'
    r'accuracy:' + FLOAT_CAPTURE + r'\s+'     # Group 1: accuracy
    r'precision:' + FLOAT_CAPTURE + r'\s+'    # Group 2: precision
    r'recall:' + FLOAT_CAPTURE + r'\s+'       # Group 3: recall
    r'f1:' + FLOAT_CAPTURE + r'\s+'           # Group 4: f1
    r'roc_auc_macro:' + FLOAT_CAPTURE + r'\s+' # Group 5: roc_auc_macro (for AUC-ROC-MACRO)
    r'roc_auc_weighted:' + r'\s+\d\.\d+' + r'\s+' # Skip roc_auc_weighted (don't capture)
    r'roc_auc_micro:' + r'\s+\d\.\d+' + r'\s+' # Skip roc_auc_micro (don't capture)
    r'pr_auc_macro:' + FLOAT_CAPTURE,         # Group 6: pr_auc_macro (for AUC-PR-MACRO)
    re.MULTILINE | re.DOTALL
)

def clean_strategy_name(strategy_folder_name):
    """Cleans the strategy name by removing the date/time/user prefix."""
    # This regex is simplified slightly but handles the core removal
    match = re.search(r'_(default_(binary|multiclass)_)?(?P<strategy_name>[^_]+)$', strategy_folder_name)
    
    if match and match.group('strategy_name'):
        cleaned = match.group('strategy_name').replace('randomundersampler', 'rundersampler')
    elif strategy_folder_name.endswith('_none'):
        cleaned = 'none'
    else:
        cleaned = strategy_folder_name

    return cleaned


def extract_metrics(filepath, model_name, strategy_name):
    """Extracts metrics based on classification type (binary or multiclass)."""
    print(f"Found: Model='{model_name}', Strategy='{strategy_name}', Path='{filepath}'")
    
    is_multiclass = '/multiclass/' in filepath
    metrics_map = MULTICLASS_METRICS if is_multiclass else BINARY_METRICS
    regex = MULTICLASS_RE if is_multiclass else BINARY_RE
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            match = regex.search(content)
            
            if not match:
                # Fallback check for binary in multiclass folder if the multiclass pattern fails
                if is_multiclass:
                    match = BINARY_RE.search(content)
                    if match:
                        print(f"  [WARN] Multiclass pattern failed, but Binary pattern matched in: {filepath}", file=sys.stderr)
                        metrics_map = BINARY_METRICS
                        is_multiclass = False
                
                if not match:
                    print(f"  [WARN] Metrics block not found (or pattern mismatch) in: {filepath}", file=sys.stderr)
                    return None

            # Map the extracted values (captured groups) to the metric keys
            extracted_metrics = {}
            for i, (table_key, file_key) in enumerate(metrics_map.items()):
                # i+1 because captured groups start at index 1
                value = float(match.group(i + 1))
                extracted_metrics[table_key] = {'str': f"{value:.4f}", 'float': value}
            
            return is_multiclass, extracted_metrics

    except Exception as e:
        print(f"  [ERROR] Could not read file {filepath}: {e}", file=sys.stderr)
        return None

def write_model_table(outfile, model, strategies, rank_key, metric_keys):
    """Helper function to write the table and ranking for one model."""
    outfile.write(f"----- {model} -----\n\n")
    
    # 1. Write the header row with proper alignment
    # "Estratégia" gets 20 chars, then each metric column gets appropriate spacing
    header_parts = ['Estratégia\t\t\t\t']  # 20 chars with padding
    for key in metric_keys:
        if len(key) <= 6:  # Short metrics like "Acur.", "Prec.", etc.
            header_parts.append(key)
        else:  # Longer metrics like "AUC-ROC-MACRO"
            header_parts.append(key)
    header = '\t\t'.join(header_parts)
    outfile.write(header + "\n")
    
    # 2. Sort by strategy name for table clarity
    sorted_strategies_for_table = sorted(strategies.items())

    for strategy, metrics in sorted_strategies_for_table:
        row = f"{strategy:<20}\t"
        # Use the formatted string ('str') values for the table
        row += "\t".join(metrics[key]['str'] for key in metric_keys)
        outfile.write(row + "\n")
    
    outfile.write("-" * 50 + "\n")

    # 3. Print Top 3 Strategies
    pr_scores = [(strategy, metrics[rank_key]['float']) for strategy, metrics in strategies.items()]
    
    # Sort by the rank key value in descending order
    ranked_strategies = sorted(pr_scores, key=operator.itemgetter(1), reverse=True)
    
    outfile.write(f"\nTOP 3 Estratégias (por {rank_key}):\n")
    
    top_3 = ranked_strategies[:3]
    
    if top_3:
        for rank, (strategy_name, pr_score) in enumerate(top_3):
            formatted_score = f"{pr_score:.4f}"
            outfile.write(f"  {rank + 1}. {strategy_name:<15} ({rank_key}: {formatted_score})\n")
    else:
         outfile.write("  N/A (Menos de 3 estratégias encontradas)\n")

    outfile.write("\n\n") # Double newline to separate models

def main():
    """Traverses the directory, collects metrics, and saves the tables sectioned by type."""
    # Data is collected into two separate dictionaries
    binary_data = defaultdict(dict)
    multiclass_data = defaultdict(dict)
    
    print(f"Searching for files named '{RESULTS_FILENAME}' in '{ROOT_DIR}'...")
    print("-" * 50)
    
    # 1. Collect Data
    for root, dirs, files in os.walk(ROOT_DIR):
        if RESULTS_FILENAME in files:
            filepath = os.path.join(root, RESULTS_FILENAME)
            
            model_name = os.path.basename(root)
            strategy_folder_name = os.path.basename(os.path.dirname(root))
            strategy_name_clean = clean_strategy_name(strategy_folder_name)

            extraction_result = extract_metrics(filepath, model_name, strategy_folder_name)
            
            if extraction_result:
                is_multiclass, metrics = extraction_result
                if is_multiclass:
                    multiclass_data[model_name][strategy_name_clean] = metrics
                else:
                    binary_data[model_name][strategy_name_clean] = metrics

    print("-" * 50)
    
    # 2. Generate and Save Tables
    
    if not (binary_data or multiclass_data):
        print("No valid metrics were found. Aborting file generation.")
        return

    with open(OUTPUT_FILENAME, 'w') as outfile:
        outfile.write("="*80 + "\n")
        outfile.write(f"AGGREGATED CLASSIFICATION RESULTS\n")
        outfile.write("="*80 + "\n\n")

        # --- Section 1: Binary Classification ---
        if binary_data:
            outfile.write("## 🗄️ BINARY CLASSIFICATION RESULTS (Ranked by AUC-PR)\n")
            outfile.write("---" + "\n\n")
            for model, strategies in binary_data.items():
                write_model_table(outfile, model, strategies, BINARY_RANK_KEY, list(BINARY_METRICS.keys()))
        else:
             outfile.write("## 🗄️ BINARY CLASSIFICATION RESULTS\n")
             outfile.write("---" + "\n")
             outfile.write("No binary results found.\n\n")

        # --- Section 2: Multiclass Classification ---
        if multiclass_data:
            outfile.write("\n" + "="*80 + "\n")
            outfile.write("## 📊 MULTICLASS CLASSIFICATION RESULTS (Ranked by AUC-PR)\n")
            outfile.write("---" + "\n\n")
            for model, strategies in multiclass_data.items():
                write_model_table(outfile, model, strategies, MULTICLASS_RANK_KEY, list(MULTICLASS_METRICS.keys()))
        else:
             outfile.write("\n" + "="*80 + "\n")
             outfile.write("## 📊 MULTICLASS CLASSIFICATION RESULTS\n")
             outfile.write("---" + "\n")
             outfile.write("No multiclass results found.\n\n")


    print("\n" + "="*80)
    print(f"✅ Success! Aggregated tables saved to: {OUTPUT_FILENAME}")
    print("="*80)


if __name__ == "__main__":
    main()