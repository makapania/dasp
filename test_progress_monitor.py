"""
Test script for the Progress Monitor

Simulates a realistic analysis workflow to demonstrate the progress monitor features.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import tkinter as tk
from spectral_predict.progress_monitor import ProgressMonitor
import random
import time


def simulate_analysis():
    """Simulate a realistic spectral analysis."""
    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()  # Hide main window

    # Create progress monitor
    total_models = 150
    monitor = ProgressMonitor(parent=root, total_models=total_models)
    monitor.show()

    # Simulate region analysis phase
    print("Phase 1: Region Analysis")
    for i in range(3):
        if monitor.is_cancelled():
            print("Analysis cancelled by user")
            return

        monitor.update({
            'stage': 'region_analysis',
            'message': f'Analyzing spectral region {i+1}/3...',
            'current': i,
            'total': total_models
        })
        root.update()
        time.sleep(0.5)

    # Simulate model testing phase
    print("\nPhase 2: Model Testing")
    models = ["PLS", "RandomForest", "PLS-DA"]
    preprocs = ["raw", "SNV", "MSC", "d1_sg7", "d2_sg7"]
    var_counts = [10, 20, 50, 100, 250, 500, 1000, "full"]
    subsets = ["full", "top100", "top250", "region1", "region2", "top3regions"]

    best_rmse = 1.0
    best_r2 = 0.0
    best_model = None

    for i in range(3, total_models):
        if monitor.is_cancelled():
            print("\nAnalysis cancelled by user")
            monitor.complete(success=False, message="Analysis cancelled by user")
            break

        # Simulate random model configuration
        model = random.choice(models)
        preproc = random.choice(preprocs)
        var_count = random.choice(var_counts)
        subset = random.choice(subsets)

        # Simulate performance metrics
        rmse = random.uniform(0.05, 0.3)
        r2 = random.uniform(0.75, 0.98)

        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_model = {
                'model': model,
                'preprocessing': preproc,
                'n_vars': var_count if isinstance(var_count, int) else 2151,
                'subset': subset,
                'task_type': 'regression',
                'RMSE': rmse,
                'R2': r2
            }

        # Update progress
        progress_data = {
            'stage': 'model_testing',
            'message': f'Testing {model} with {preproc} preprocessing ({var_count} vars, {subset})',
            'current': i,
            'total': total_models,
            'best_model': best_model
        }

        monitor.update(progress_data)
        root.update()

        # Simulate processing time (faster for demo)
        time.sleep(0.05)

        # Print progress every 10 models
        if i % 10 == 0:
            print(f"[{i}/{total_models}] Best RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}")

    if not monitor.is_cancelled():
        print("\nAnalysis Complete!")
        print(f"Final Best Model: {best_model['model']} with {best_model['preprocessing']}")
        print(f"Performance: RMSE={best_rmse:.4f}, R²={best_r2:.4f}")

        monitor.complete(
            success=True,
            message=f"All {total_models} models tested successfully! Best RMSE: {best_rmse:.4f}"
        )

    # Keep window open
    root.mainloop()


def test_classification_monitor():
    """Test progress monitor with classification metrics."""
    root = tk.Tk()
    root.withdraw()

    total_models = 100
    monitor = ProgressMonitor(parent=root, total_models=total_models)
    monitor.show()

    models = ["PLS-DA", "RandomForest"]
    preprocs = ["raw", "SNV", "d1_sg7"]

    best_roc = 0.0
    best_acc = 0.0
    best_model = None

    for i in range(total_models):
        if monitor.is_cancelled():
            monitor.complete(success=False, message="Cancelled")
            break

        model = random.choice(models)
        preproc = random.choice(preprocs)

        roc_auc = random.uniform(0.85, 0.99)
        accuracy = random.uniform(0.80, 0.95)

        if roc_auc > best_roc:
            best_roc = roc_auc
            best_acc = accuracy
            best_model = {
                'model': model,
                'preprocessing': preproc,
                'n_vars': random.randint(50, 500),
                'subset': 'full',
                'task_type': 'classification',
                'ROC_AUC': roc_auc,
                'Accuracy': accuracy
            }

        monitor.update({
            'stage': 'model_testing',
            'message': f'Testing {model} with {preproc}',
            'current': i + 1,
            'total': total_models,
            'best_model': best_model
        })

        root.update()
        time.sleep(0.05)

    if not monitor.is_cancelled():
        monitor.complete(
            success=True,
            message=f"Classification complete! Best ROC AUC: {best_roc:.4f}"
        )

    root.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("Progress Monitor Demo")
    print("=" * 60)
    print()
    print("This demo simulates a realistic spectral analysis workflow")
    print("with ~150 model configurations.")
    print()
    print("Features demonstrated:")
    print("  • Real-time progress bar and percentage")
    print("  • Current model being tested")
    print("  • Best model tracking (updates when better model found)")
    print("  • Elapsed time counter")
    print("  • ETA calculation")
    print("  • Cancel button")
    print()

    choice = input("Choose demo: [1] Regression (default), [2] Classification: ").strip()

    if choice == "2":
        print("\nRunning classification demo...\n")
        test_classification_monitor()
    else:
        print("\nRunning regression demo...\n")
        simulate_analysis()
