# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # For handling cv_results_
import time
import os
import warnings
from typing import List, Dict, Tuple, Any, Optional

from sklearn.base import clone, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning # To specifically catch convergence warnings

# === Function Definition ===

def tune_hyperparameters(
    base_pipeline: Pipeline,
    param_dist: Dict[str, Any], # Distribution or list of parameters
    X_vec: np.ndarray,
    Y: np.ndarray,
    task_names: List[str],
    n_splits: int,
    n_tuning_iter: int,
    random_state: int
    ) -> Tuple[
        Dict[str, Optional[BaseEstimator]], # Best fitted estimators per task
        Dict[str, Optional[Dict]],          # Best parameters per task
        Dict[str, plt.Figure]               # Tuning plots per task (Score vs Alpha)
    ]:

    print("\n--- Hyperparameter Tuning (Randomized Search CV) ---")

    # Initialize dictionaries to store results
    best_estimators_per_task: Dict[str, Optional[BaseEstimator]] = {}
    best_params_per_task: Dict[str, Optional[Dict]] = {}
    tuning_figures: Dict[str, plt.Figure] = {} # To store generated plots
    total_convergence_warnings = 0 # Counter for warnings across all tasks

    # --- Task Loop ---
    for task_index, task_name in enumerate(task_names):
        print(f"\nTuning for Task {task_index + 1}/{len(task_names)}: {task_name}")
        y_task = Y[:, task_index] # Binary target for this task
        fig_tuning = None # Initialize figure variable for this task's plot
        task_convergence_warnings = 0 # Counter for this task

        # Check for single class issue
        if len(np.unique(y_task)) < 2:
            print(f"    ⚠️ Skipping tuning for task {task_name}: Only one class present.")
            best_estimators_per_task[task_name] = None
            best_params_per_task[task_name] = None
            continue # Skip to next task

        # Setup CV for tuning this binary task
        tuning_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Instantiate RandomizedSearchCV
        # Clone base_pipeline to ensure each search starts fresh
        random_search = RandomizedSearchCV(
            estimator=clone(base_pipeline),
            param_distributions=param_dist,
            n_iter=n_tuning_iter,
            scoring='roc_auc', # Using AUC for tuning is generally robust
            cv=tuning_cv,
            n_jobs=-1,       # Use parallel processing if available
            verbose=1,       # Set verbosity level (1, 2, or higher)
            random_state=random_state,
            error_score='raise' # Raise errors during tuning fits
        )

        try:
            print(f"    Fitting RandomizedSearchCV (n_iter={n_tuning_iter})...")
            # --- Catch Warnings during fit ---
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                # Filter specifically for ConvergenceWarning
                warnings.simplefilter("always", ConvergenceWarning)
                # Fit the model
                random_search.fit(X_vec, y_task)

                # Check for convergence warnings specifically
                for warning_message in w:
                    if issubclass(warning_message.category, ConvergenceWarning):
                        task_convergence_warnings += 1
            # --- End Catch Warnings ---

            # Store best results
            best_estimators_per_task[task_name] = random_search.best_estimator_
            best_params_per_task[task_name] = random_search.best_params_
            best_score = random_search.best_score_

            print(f"    ✅ Best AUC: {best_score:.4f}")
            print(f"    ✅ Best Params: {best_params_per_task[task_name]}")
            if task_convergence_warnings > 0:
                print(f"    ⚠️ Encountered {task_convergence_warnings} ConvergenceWarning(s) during tuning for {task_name}.")
                total_convergence_warnings += task_convergence_warnings

            # --- Generate Score vs Alpha Plot ---
            # This plot is most meaningful if 'estimator__alpha' is in param_dist
            alpha_key = 'estimator__alpha' # Key for alpha in param_dist
            if alpha_key in random_search.cv_results_.get('params', [{}])[0]: # Check if alpha was tuned
                print(f"    Generating tuning plot (Score vs Alpha) for {task_name}...")
                try:
                    cv_results = pd.DataFrame(random_search.cv_results_)
                    # Extract alpha values and scores
                    # Ensure correct key is used based on your param_dist
                    alphas = cv_results[f'param_{alpha_key}'].astype(float)
                    mean_scores = cv_results['mean_test_score']
                    std_scores = cv_results['std_test_score'] # For error bars

                    fig_tuning, ax = plt.subplots(figsize=(8, 5))
                    # Use errorbar for better visualization
                    # Sort by alpha for a cleaner line plot if desired, though scatter shows samples better
                    sorted_indices = np.argsort(alphas)
                    ax.errorbar(alphas[sorted_indices], mean_scores[sorted_indices],
                                yerr=std_scores[sorted_indices], fmt='o--', # Plot points and connecting line
                                capsize=4, alpha=0.7, label='CV Score per Sampled Alpha')

                    # Highlight the best point
                    best_alpha = best_params_per_task[task_name][alpha_key]
                    ax.scatter([best_alpha], [best_score], color='red', s=100, zorder=5,
                               label=f'Best Alpha ({best_alpha:.4g})') # Use general format for alpha

                    ax.set_xscale('log') # Alpha spans orders of magnitude
                    ax.set_xlabel('Estimator Alpha (log scale)')
                    ax.set_ylabel('Mean CV ROC AUC Score')
                    ax.set_title(f'Tuning Results: {task_name}')
                    ax.legend()
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    plt.tight_layout()

                    # Store the figure using a descriptive key
                    safe_task_name = "".join(c if c.isalnum() else "_" for c in task_name)
                    tuning_figures[f"tuning_{safe_task_name}"] = fig_tuning
                    print(f"    Tuning plot figure created for {task_name}.")

                except KeyError as ke:
                    print(f"    ⚠️ Could not generate tuning plot: Missing key in cv_results_ ({ke}). Check param_dist keys.")
                    if fig_tuning: plt.close(fig_tuning)
                except Exception as plot_e:
                    print(f"    ⚠️ Error generating tuning plot for {task_name}: {plot_e}")
                    if fig_tuning: plt.close(fig_tuning) # Close if created but failed later
            else:
                print(f"    Skipping tuning plot generation: '{alpha_key}' not found in tuned parameters.")
            # --- END Generate Plot ---

        except Exception as e:
            print(f"    ⚠️ Error during tuning for task {task_name}: {e}")
            best_estimators_per_task[task_name] = None
            best_params_per_task[task_name] = None
            if fig_tuning: plt.close(fig_tuning) # Close plot if it was started but tuning failed

    # --- Final Summary ---
    print(f"\n--- Hyperparameter Tuning Complete ---")
    if total_convergence_warnings > 0:
         print(f"⚠️ Total ConvergenceWarnings across all tasks during tuning: {total_convergence_warnings}")
         print(f"   Consider increasing max_iter in MLPClassifier or adjusting learning rate/solver if this persists.")

    # Return the collected results and figures
    return best_estimators_per_task, best_params_per_task, tuning_figures