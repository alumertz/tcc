#!/usr/bin/env python3
"""
Script to update results.json with the best holdout results from compiled_best_holdout_results.txt
"""

import json
import re
from pathlib import Path


def parse_compiled_results(filepath):
    """Parse the compiled_best_holdout_results.txt file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by model sections
    model_sections = re.split(r'={80}\nMODEL: ([A-Z_]+)\n={80}', content)[1:]  # Skip header
    
    results = {}
    
    # Process pairs of (model_name, section_content)
    for i in range(0, len(model_sections), 2):
        model_name = model_sections[i].lower()
        section_content = model_sections[i + 1]
        
        # Split by individual results
        result_blocks = re.split(r'--- Result \d+/\d+ ---', section_content)[1:]  # Skip empty first
        
        for block in result_blocks:
            # Extract experiment name
            exp_match = re.search(r'Experiment: (.+)', block)
            if not exp_match:
                continue
            experiment = exp_match.group(1)
            
            # Determine classification type from experiment name
            if 'binary' in experiment.lower():
                classification_type = 'binary'
            elif 'multiclass' in experiment.lower():
                classification_type = 'multiclass'
            else:
                continue
            
            # Extract balancing strategy
            balance_match = re.search(r'Balancing Strategy: (\w+)', block)
            if not balance_match:
                continue
            balancing = balance_match.group(1).lower()
            
            # Extract parameters
            params_match = re.search(r'Parameters: (\{.*?\})', block)
            if not params_match:
                continue
            params_str = params_match.group(1)
            # Convert string to dict
            import ast
            params = ast.literal_eval(params_str)
            
            # Extract PR-AUC
            pr_auc_match = re.search(r'PR-AUC\.*\s+([\d.]+)', block)
            if not pr_auc_match:
                continue
            pr_auc = float(pr_auc_match.group(1))
            
            # Store result
            key = (model_name, classification_type)
            if key not in results or pr_auc > results[key]['pr_auc']:
                results[key] = {
                    'model': model_name,
                    'type': classification_type,
                    'experiment': experiment,
                    'balancing': balancing,
                    'hyperparameters': params,
                    'pr_auc': pr_auc
                }
    
    return results


def main():
    compiled_file = Path('/Users/i583975/git/tcc/results_final_tcc/compiled_best_holdout_results.txt')
    results_json_file = Path('/Users/i583975/git/tcc/results.json')
    
    print("="*80)
    print("UPDATING results.json WITH BEST HOLDOUT RESULTS")
    print("="*80)
    
    # Parse compiled results
    print(f"\nParsing: {compiled_file}")
    best_results = parse_compiled_results(compiled_file)
    
    print(f"\nFound best results for {len(best_results)} model/type combinations:")
    for key, result in sorted(best_results.items()):
        model, clf_type = key
        print(f"  {clf_type:10s} | {model:30s} | PR-AUC: {result['pr_auc']:.4f} | Exp: {result['experiment']}")
    
    # Convert to list format for JSON
    results_list = []
    for (model, classification_type), result in sorted(best_results.items()):
        # Remove pr_auc before saving to JSON
        result_copy = {
            'type': result['type'],
            'model': result['model'],
            'experiment': result['experiment'],
            'balancing': result['balancing'],
            'hyperparameters': result['hyperparameters']
        }
        results_list.append(result_copy)
    
    # Save to results.json
    print(f"\nSaving to: {results_json_file}")
    with open(results_json_file, 'w') as f:
        json.dump(results_list, f, indent=4)
    
    print(f"\n✓ Updated results.json with {len(results_list)} best configurations")
    print("\nBreakdown:")
    binary_count = sum(1 for r in results_list if r['type'] == 'binary')
    multiclass_count = sum(1 for r in results_list if r['type'] == 'multiclass')
    print(f"  Binary: {binary_count}")
    print(f"  Multiclass: {multiclass_count}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
