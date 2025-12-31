#!/usr/bin/env python3
"""
Small script to test Optuna visualization plots.
Creates a simple optimization study and saves all visualization plots.
"""

import optuna
import optuna.visualization as vis
from pathlib import Path


def objective(trial):
    """Simple objective function for testing."""
    # Suggest some hyperparameters
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    
    # Simple function to optimize (Rosenbrock function)
    return (1 - x)**2 + 100 * (y - x**2)**2 + learning_rate * n_estimators - max_depth


def main():
    print("Creating Optuna study...")
    
    # Create study
    study = optuna.create_study(direction='minimize', study_name='test_study')
    
    # Run optimization
    print("Running optimization with 50 trials...")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Create output directory
    output_dir = Path("./test_optuna_plots")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving visualization plots to {output_dir}...")
    
    try:
        # Save optimization history
        print("  - Saving optimization history...")
        fig = vis.plot_optimization_history(study)
        fig.write_image(str(output_dir / "optimization_history.png"))
        
        # Save parameter importances
        print("  - Saving parameter importances...")
        fig = vis.plot_param_importances(study)
        fig.write_image(str(output_dir / "param_importances.png"))
        
        # Save parallel coordinate plot
        print("  - Saving parallel coordinate plot...")
        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(str(output_dir / "parallel_coordinate.png"))
        
        # Save slice plot
        print("  - Saving slice plot...")
        fig = vis.plot_slice(study)
        fig.write_image(str(output_dir / "slice_plot.png"))
        
        # Save contour plot (2D)
        print("  - Saving contour plot...")
        fig = vis.plot_contour(study, params=['x', 'y'])
        fig.write_image(str(output_dir / "contour_plot.png"))
        
        print(f"\n✓ All plots saved successfully to {output_dir}")
        print("\nPlots saved as PNG images.")
        
    except Exception as e:
        print(f"\n✗ Error saving plots: {e}")


if __name__ == "__main__":
    main()
