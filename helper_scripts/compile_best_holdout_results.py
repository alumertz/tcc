"""
Script to compile OVERALL BEST COMBINATION sections from all holdout_results42.txt files
into a single organized file, grouped by model.

Author: Generated for TCC project
Date: December 2025
"""

from pathlib import Path
import re


def extract_best_combination(file_path):
    """
    Extract the OVERALL BEST COMBINATION section from a holdout_results42.txt file.
    
    Args:
        file_path: Path to the holdout_results42.txt file
        
    Returns:
        Dictionary with fold, strategy, parameters, and performance metrics
        Returns None if section not found
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the OVERALL BEST COMBINATION section
        pattern = r'={80}\s*OVERALL BEST COMBINATION:\s*={80}\s*(.*?)(?=\n={80}|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return None
        
        section = match.group(1).strip()
        
        # Extract components
        result = {}
        
        # Extract fold
        fold_match = re.search(r'Fold:\s*(\d+)', section)
        if fold_match:
            result['fold'] = fold_match.group(1)
        
        # Extract balancing strategy
        strategy_match = re.search(r'Balancing Strategy:\s*(\w+)', section)
        if strategy_match:
            result['strategy'] = strategy_match.group(1)
        
        # Extract parameters
        params_match = re.search(r'Parameters:\s*(\{.*?\})', section)
        if params_match:
            result['parameters'] = params_match.group(1)
        
        # Extract performance metrics
        metrics = {}
        metric_pattern = r'(Accuracy|Precision|Recall|F1-Score|ROC-AUC|PR-AUC):\s*([\d.]+)'
        for match in re.finditer(metric_pattern, section):
            metrics[match.group(1)] = match.group(2)
        
        result['metrics'] = metrics
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def get_experiment_name(path):
    """
    Extract the experiment name from the path.
    
    Args:
        path: Path object pointing to the experiment folder
        
    Returns:
        Experiment name as string
    """
    # The experiment folder is the parent of the model folder
    # e.g., results_final_tcc/_optimized_binary_first_20251203_215554/decision_tree/
    # We want "_optimized_binary_first_20251203_215554"
    experiment_folder = path.parent.name
    return experiment_folder


def compile_best_results(base_path, output_file):
    """
    Compile all OVERALL BEST COMBINATION sections from holdout_results42.txt files.
    
    Args:
        base_path: Base directory containing experiment folders
        output_file: Path to output file
    """
    base_path = Path(base_path)
    
    # Dictionary to store results grouped by model
    model_results = {}
    
    # Find all holdout_results42.txt files
    for results_file in base_path.rglob('holdout_results42.txt'):
        # Get model name (parent directory name)
        model_name = results_file.parent.name
        
        # Get experiment name
        experiment_name = get_experiment_name(results_file.parent)
        
        # Extract best combination
        best_combo = extract_best_combination(results_file)
        
        if best_combo:
            # Initialize model entry if needed
            if model_name not in model_results:
                model_results[model_name] = []
            
            # Add experiment name to the result
            best_combo['experiment'] = experiment_name
            best_combo['full_path'] = str(results_file.relative_to(base_path))
            
            model_results[model_name].append(best_combo)
            print(f"Processed: {experiment_name}/{model_name}")
    
    # Write compiled results
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPILED BEST HOLDOUT RESULTS (with random_seed=42)\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nTotal models found: {len(model_results)}\n")
        f.write(f"Total results compiled: {sum(len(v) for v in model_results.values())}\n")
        f.write("\n")
        
        # Sort models alphabetically
        for model_name in sorted(model_results.keys()):
            f.write("=" * 80 + "\n")
            f.write(f"MODEL: {model_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            results = model_results[model_name]
            
            # Sort by experiment name for consistent ordering
            results.sort(key=lambda x: x['experiment'])
            
            for idx, result in enumerate(results, 1):
                f.write(f"--- Result {idx}/{len(results)} ---\n")
                f.write(f"Experiment: {result['experiment']}\n")
                f.write(f"File: {result['full_path']}\n")
                f.write(f"\n")
                f.write(f"Best Fold: {result.get('fold', 'N/A')}\n")
                f.write(f"Balancing Strategy: {result.get('strategy', 'N/A')}\n")
                f.write(f"Parameters: {result.get('parameters', 'N/A')}\n")
                f.write(f"\n")
                f.write(f"Performance:\n")
                
                metrics = result.get('metrics', {})
                for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']:
                    if metric_name in metrics:
                        f.write(f"  {metric_name:.<15} {metrics[metric_name]}\n")
                
                f.write("\n")
                f.write("-" * 80 + "\n\n")
            
            f.write("\n")
    
    print(f"\n✓ Compilation complete!")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Models processed: {len(model_results)}")
    print(f"✓ Total results: {sum(len(v) for v in model_results.values())}")


def main():
    """Main execution function."""
    # Configuration
    base_path = Path("results_final_tcc")
    output_file = base_path / "compiled_best_holdout_results.txt"
    
    print("=" * 80)
    print("COMPILING BEST HOLDOUT RESULTS")
    print("=" * 80)
    print(f"\nSearching in: {base_path.absolute()}")
    print(f"Output file: {output_file.absolute()}\n")
    
    # Check if base path exists
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist!")
        return
    
    # Compile results
    compile_best_results(base_path, output_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
