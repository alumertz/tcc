#!/usr/bin/env python3
"""
Script to extract best parameters from holdout_results.txt files across experiments.
Organizes parameters by model and experiment.
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def extract_best_params_from_file(filepath):
    """Extract the 5 'Best Parameters:' lines and their PR-AUC values from a holdout_results.txt file."""
    params_with_metrics = []
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Split content into fold sections
        fold_sections = re.split(r'Fold \d+:', content)
        
        for section in fold_sections[1:]:  # Skip the header
            # Extract best parameters
            params_match = re.search(r'Best Parameters: (\{[^}]+\})', section)
            # Extract PR-AUC value
            pr_auc_match = re.search(r'PR-AUC: ([\d.]+)', section)
            
            if params_match and pr_auc_match:
                params = params_match.group(1)
                pr_auc = pr_auc_match.group(1)
                params_with_metrics.append((pr_auc, params))
        
        # Take the first 5 matches (one for each fold)
        params_with_metrics = params_with_metrics[:5]
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return params_with_metrics


def scan_experiments(base_dir):
    """
    Scan the optimized_binary directory for experiments and extract best parameters.
    Returns a dictionary organized by model -> experiment -> parameters.
    """
    results = defaultdict(lambda: defaultdict(list))
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return results
    
    # Iterate through experiment folders (e.g., 20251123_235823_ana_optimized_binary_none)
    for experiment_dir in sorted(base_path.iterdir()):
        if not experiment_dir.is_dir():
            continue
        
        experiment_name = experiment_dir.name
        
        # Iterate through model folders (e.g., catboost, decision_tree, etc.)
        for model_dir in sorted(experiment_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            # Look for holdout_results.txt
            holdout_file = model_dir / "holdout_results.txt"
            
            if holdout_file.exists():
                params = extract_best_params_from_file(holdout_file)
                if params:
                    results[model_name][experiment_name] = params
                    print(f"Found {len(params)} parameters in {model_name}/{experiment_name}")
    
    return results


def write_results_to_file(results, output_file):
    """Write the extracted parameters to a structured text file."""
    with open(output_file, 'w') as f:
        # Sort models alphabetically
        for model_name in sorted(results.keys()):
            f.write(f"{model_name}:\n\n")
            
            # Sort experiments alphabetically
            for experiment_name in sorted(results[model_name].keys()):
                f.write(f"experiment_{experiment_name}\n")
                
                params_list = results[model_name][experiment_name]
                for i, (pr_auc, params) in enumerate(params_list, 1):
                    f.write(f"{i} - PR-AUC: {pr_auc}, params: {params}\n")
                
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"\nResults written to: {output_file}")


def list_available_folders(results_dir):
    """List all available folders in the results directory."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return []
    
    folders = [f.name for f in results_path.iterdir() if f.is_dir()]
    return sorted(folders)


def main():
    # Base results directory
    results_dir = "/Users/i583975/git/tcc/results"
    
    # List available folders
    print("Available folders in /results:")
    print("-" * 50)
    
    available_folders = list_available_folders(results_dir)
    
    if not available_folders:
        print("No folders found in results directory.")
        return
    
    for idx, folder in enumerate(available_folders, 1):
        print(f"{idx}. {folder}")
    
    print("-" * 50)
    
    # Get user choice
    try:
        choice = input("\nEnter the number of the folder to scan (or press Enter for 'optimized_binary'): ").strip()
        
        if choice == "":
            selected_folder = "optimized_binary"
            print(f"Using default: {selected_folder}")
        else:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_folders):
                selected_folder = available_folders[choice_idx]
            else:
                print("Invalid choice. Using default 'optimized_binary'")
                selected_folder = "optimized_binary"
    except ValueError:
        print("Invalid input. Using default 'optimized_binary'")
        selected_folder = "optimized_binary"
    
    # Set paths
    base_dir = os.path.join(results_dir, selected_folder)
    output_file = os.path.join(results_dir, selected_folder, "all_best_params.txt")
    
    print(f"\nScanning experiments in: {selected_folder}")
    print("=" * 50)
    results = scan_experiments(base_dir)
    
    if results:
        write_results_to_file(results, output_file)
        print(f"\nExtracted parameters from {len(results)} models")
        for model_name, experiments in results.items():
            print(f"  {model_name}: {len(experiments)} experiments")
    else:
        print("No results found.")


if __name__ == "__main__":
    main()
