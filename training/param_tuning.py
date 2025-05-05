# === Imports ===
# (Ensure these are imported in the file where tune_hyperparameters is defined)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Useful for handling cv_results_
from typing import List, Dict, Tuple, Any, Optional
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# === Function Definition ===

# MODIFY return type to include Dict[str, plt.Figure] for plots
def tune_hyperparameters(base_pipeline: Pipeline, param_dist: Dict, X_vec: np.ndarray, Y: np.ndarray,
                         task_names: List[str], n_splits: int, n_tuning_iter: int, random_state: int
                         ) -> Tuple[
                             Dict[str, Optional[BaseEstimator]], # Best estimators
                             Dict[str, Optional[Dict]],          # Best params
                             Dict[str, plt.Figure]               # Tuning plots <<-- ADDED
                         ]:
    """Performs RandomizedSearchCV for each task and generates score vs alpha plots."""
    print("\n--- Hyperparameter Tuning ---")
    best_estimators_per_task: Dict[str, Optional[BaseEstimator]] = {}
    best_params_per_task: Dict[str, Optional[Dict]] = {}
    tuning_figures: Dict[str, plt.Figure] = {} # <-- Initialize dict for plots

    for task_index, task_name in enumerate(task_names):
        print(f"\nTuning for Task {task_index + 1}/{len(task_names)}: {task_name}")
        y_task = Y[:, task_index]
        fig_tuning = None # Initialize figure variable for this task

        if len(np.unique(y_task)) < 2:
            print(f"    ⚠️ Skipping tuning: Only one class present.")
            best_estimators_per_task[task_name] = None
            best_params_per_task[task_name] = None
            continue # Skip to next task

        tuning_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        random_search = RandomizedSearchCV(
            estimator=clone(base_pipeline),
            param_distributions=param_dist,
            n_iter=n_tuning_iter,
            scoring='roc_auc',
            cv=tuning_cv,
            n_jobs=-1,
            verbose=1,
            random_state=random_state,
            error_score='raise'
        )

        try:
            random_search.fit(X_vec, y_task)
            best_estimators_per_task[task_name] = random_search.best_estimator_
            best_params_per_task[task_name] = random_search.best_params_
            best_score = random_search.best_score_
            print(f"    ✅ Best AUC: {best_score:.4f}, Params: {best_params_per_task[task_name]}")

            # --- ADDED: Generate Score vs Alpha Plot ---
            print(f"    Generating tuning plot for {task_name}...")
            try:
                cv_results = pd.DataFrame(random_search.cv_results_)
                # Extract alpha values and scores
                alphas = cv_results['param_estimator__alpha'].astype(float)
                mean_scores = cv_results['mean_test_score']
                std_scores = cv_results['std_test_score'] # For error bars

                fig_tuning, ax = plt.subplots(figsize=(8, 5))
                # Use errorbar for better visualization
                ax.errorbar(alphas, mean_scores, yerr=std_scores, fmt='o', capsize=4, alpha=0.6, label='CV Score per Sampled Alpha')
                # Highlight the best point
                best_alpha = best_params_per_task[task_name]['estimator__alpha']
                ax.scatter([best_alpha], [best_score], color='red', s=100, zorder=5, label=f'Best Alpha ({best_alpha:.3f})')

                ax.set_xscale('log') # Alpha spans orders of magnitude
                ax.set_xlabel('Ridge Alpha (log scale)')
                ax.set_ylabel('Mean CV ROC AUC Score')
                ax.set_title(f'Tuning Results: {task_name}')
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()

                # Store the figure using a descriptive key
                safe_task_name = "".join(c if c.isalnum() else "_" for c in task_name)
                tuning_figures[f"tuning_{safe_task_name}"] = fig_tuning
                print(f"    Tuning plot figure created for {task_name}.")

            except Exception as plot_e:
                print(f"    ⚠️ Error generating tuning plot for {task_name}: {plot_e}")
                if fig_tuning: plt.close(fig_tuning) # Close if created but failed later
            # --- END ADDED ---

        except Exception as e:
            print(f"    ⚠️ Error during tuning: {e}")
            best_estimators_per_task[task_name] = None
            best_params_per_task[task_name] = None
            if fig_tuning: plt.close(fig_tuning) # Close if plot started but tuning failed

    # Report summary (optional here, as it's printed in the loop)
    print("\n--- Tuning Complete ---") # Simplified message

    # MODIFY return statement
    return best_estimators_per_task, best_params_per_task, tuning_figures