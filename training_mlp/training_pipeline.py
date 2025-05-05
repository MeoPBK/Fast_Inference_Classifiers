# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- MODIFIED: Import MLPClassifier ---
from sklearn.neural_network import MLPClassifier
# --- END MODIFICATION ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Use loguniform for alpha, learning_rate_init; randint for layers; choice for activation/solver
from scipy.stats import loguniform, randint
from random import choice # Alternative way to specify choices

# Assuming your custom modules are structured correctly and importable
try:
    from save_and_load import save_results, load_and_prepare_data
    from train_bootstrap import train_evaluate_bootstrap
    from param_tuning import tune_hyperparameters
    from learning_curve.run_analysis import run_diagnostic_analysis
    from post_analysis import run_post_training_analysis
    from threshold_tuning import find_optimal_thresholds
    MODULES_LOADED = True
except ImportError as ie:
    print(f"Error importing required modules: {ie}")
    MODULES_LOADED = False
    exit()

import warnings # Add warnings module
from sklearn.exceptions import ConvergenceWarning # To specifically catch convergence warnings
# === Configuration Constants ===
TASK_NAMES = [
    "Retrieve",
    "Internet",
    "ChatMemory",
    "RunCode"
]
# Reduce splits/iterations/bootstraps for faster testing/debugging.
# Increase them for final, more robust training runs.

# N_SPLITS = 5
# N_BOOTSTRAP = 10
# N_TUNING_ITER = 30 # Reduced iterations for MLP tuning (can be slow)
N_SPLITS = 3 # Reduced from 5 for faster CV
N_BOOTSTRAP = 5 # Reduced from 10 for faster final ensemble build
N_TUNING_ITER = 20 # Reduced from 50/30 as MLP tuning is expensive
RANDOM_STATE = 42
# --- MODIFIED: Output filename ---
MODEL_FILENAME = "./Classifier_Multilabel_Light/trained_models/model_multilabel_MLP_tuned_bootstrapped.pkl"
# --- END MODIFICATION ---
DATA_PATH = "./Classifier_Multilabel_Light/training_data/training_data_07.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PLOTS_DIR = "plots_mlp" # Use a different directory for MLP plots

# === Main Execution ===
if __name__ == "__main__":
    if not MODULES_LOADED:
        print("Exiting due to missing modules.")
        exit()

    start_time = time.time()

    # 1. Load Data
    X_vec, Y, vectorizer = load_and_prepare_data(DATA_PATH, TASK_NAMES, RANDOM_STATE, EMBEDDING_MODEL)
    if X_vec is None or Y is None or vectorizer is None: exit()

    # 2. Define Base Pipeline
    # --- MODIFIED: Use MLPClassifier ---
    print("\n--- Defining Base MLP Pipeline ---")
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', MLPClassifier(
            # Sensible defaults, some will be tuned
            activation='relu',        # Common activation
            solver='adam',            # Popular optimizer
            early_stopping=True,      # Stop if validation score doesn't improve
            validation_fraction=0.1,  # Use 10% of training data for early stopping validation
            n_iter_no_change=10,      # How many epochs without improvement to stop
            max_iter=500,             # Increased max iterations
            random_state=RANDOM_STATE
        ))
    ])
    print(f"Base Pipeline: {base_pipeline}")
    # --- END MODIFICATION ---

    # 3. Diagnostic Analysis (Optional but Recommended)
    # Note: Multi-label curve might still fail here depending on sklearn version/interactions
    _ = run_diagnostic_analysis(X_vec, Y, TASK_NAMES, base_pipeline, N_SPLITS, RANDOM_STATE)

    # 4. Tune Hyperparameters
    # --- MODIFIED: Define MLP parameter distributions ---
    print("\n--- Defining MLP Hyperparameter Search Space ---")
    param_dist = {
        # Hidden Layers: e.g., one layer 50-150 neurons, or two layers
        'estimator__hidden_layer_sizes': [(s,) for s in randint(50, 151).rvs(10)] + \
                                         [(s1, s2) for s1, s2 in zip(randint(30, 101).rvs(10), randint(10, 51).rvs(10))], # Sample some 1 & 2 layer configs
        'estimator__alpha': loguniform(1e-5, 1e-1), # L2 regularization (smaller range often needed for MLP)
        'estimator__learning_rate_init': loguniform(1e-4, 1e-2), # Initial learning rate for adam/sgd
        # Add others if desired, e.g.:
        # 'estimator__activation': ['relu', 'tanh'],
        # 'estimator__solver': ['adam', 'sgd'],
        # 'estimator__batch_size': [32, 64, 128],
    }
    print(f"Parameter distributions for tuning: {param_dist}")
    # --- END MODIFICATION ---

    best_estimators, best_params, tuning_figs = tune_hyperparameters( # Assuming tune_hyperparameters returns figures
        base_pipeline, param_dist, X_vec, Y, TASK_NAMES, N_SPLITS, N_TUNING_ITER, RANDOM_STATE
    )
    # Add tuning figures to collection (if tune_hyperparameters was modified to return them)
    all_figures_to_save = {}
    all_figures_to_save.update(tuning_figs)


    # 5. Train, Evaluate (CV), and Bootstrap Ensemble
    # --- MODIFIED: train_evaluate_bootstrap needs internal changes for predict_proba ---
    # The call signature remains the same, but the function's internals must be updated
    (ensemble_models,
     mean_cv_test_scores_auc, mean_cv_train_scores_auc,
     mean_cv_test_scores_f1, mean_cv_train_scores_f1,
     mean_cv_test_scores_bal_acc, mean_cv_train_scores_bal_acc,
     mean_cv_test_scores_mcc, mean_cv_train_scores_mcc,
     oof_predictions # This should now contain probabilities
     ) = train_evaluate_bootstrap(best_estimators,
                                  X_vec, Y, TASK_NAMES,
                                  N_SPLITS, N_BOOTSTRAP, RANDOM_STATE
                                 )
    # --- END MODIFICATION ---

    # 6. Find Optimal Thresholds
    # --- MODIFIED: find_optimal_thresholds needs internal changes for probabilities ---
    # The call signature remains the same, but the function's internals must be updated
    # The default threshold inside should likely be 0.5 now
    optimal_thresholds = find_optimal_thresholds(oof_predictions, TASK_NAMES, default_threshold=0.5)
    # --- END MODIFICATION ---

    # 7. Print final evaluation scores
    # (Printing logic remains the same)
    print("\n--- Final Mean CV Scores ---")
    print("\n" + "="*40)
    print("--- Final Mean CV Scores (Across Folds) ---")
    print("="*40)
    for task_name in TASK_NAMES:
        print(f"\n  === Task: {task_name} ===")

        # Retrieve scores safely using .get()
        mean_auc_test = mean_cv_test_scores_auc.get(task_name, np.nan)
        mean_auc_train = mean_cv_train_scores_auc.get(task_name, np.nan)
        mean_f1_test = mean_cv_test_scores_f1.get(task_name, np.nan)
        mean_f1_train = mean_cv_train_scores_f1.get(task_name, np.nan)
        mean_bal_acc_test = mean_cv_test_scores_bal_acc.get(task_name, np.nan)
        mean_bal_acc_train = mean_cv_train_scores_bal_acc.get(task_name, np.nan)
        mean_mcc_test = mean_cv_test_scores_mcc.get(task_name, np.nan)
        mean_mcc_train = mean_cv_train_scores_mcc.get(task_name, np.nan)

        # --- Print AUC ---
        print(f"    Mean Test AUC        : {mean_auc_test:.4f}" if not np.isnan(mean_auc_test) else "    Mean Test AUC        : N/A")
        print(f"    Mean Train AUC       : {mean_auc_train:.4f}" if not np.isnan(mean_auc_train) else "    Mean Train AUC       : N/A")
        if not np.isnan(mean_auc_train) and not np.isnan(mean_auc_test):
                print(f"    AUC Train-Test Gap   : {mean_auc_train - mean_auc_test:.4f}")
        else:
                print(f"    AUC Train-Test Gap   : N/A")

        # --- Print F1 Score ---
        print(f"    Mean Test F1         : {mean_f1_test:.4f}" if not np.isnan(mean_f1_test) else "    Mean Test F1         : N/A")
        print(f"    Mean Train F1        : {mean_f1_train:.4f}" if not np.isnan(mean_f1_train) else "    Mean Train F1        : N/A")
        if not np.isnan(mean_f1_train) and not np.isnan(mean_f1_test):
                print(f"    F1 Train-Test Gap    : {mean_f1_train - mean_f1_test:.4f}")
        else:
                print(f"    F1 Train-Test Gap    : N/A")

        # --- Print Balanced Accuracy ---
        print(f"    Mean Test Bal Acc    : {mean_bal_acc_test:.4f}" if not np.isnan(mean_bal_acc_test) else "    Mean Test Bal Acc    : N/A")
        print(f"    Mean Train Bal Acc   : {mean_bal_acc_train:.4f}" if not np.isnan(mean_bal_acc_train) else "    Mean Train Bal Acc   : N/A")
        if not np.isnan(mean_bal_acc_train) and not np.isnan(mean_bal_acc_test):
                print(f"    Bal Acc Train-Test Gap: {mean_bal_acc_train - mean_bal_acc_test:.4f}")
        else:
                print(f"    Bal Acc Train-Test Gap: N/A")

        # --- Print MCC ---
        print(f"    Mean Test MCC        : {mean_mcc_test:.4f}" if not np.isnan(mean_mcc_test) else "    Mean Test MCC        : N/A")
        print(f"    Mean Train MCC       : {mean_mcc_train:.4f}" if not np.isnan(mean_mcc_train) else "    Mean Train MCC       : N/A")
        if not np.isnan(mean_mcc_train) and not np.isnan(mean_mcc_test):
                print(f"    MCC Train-Test Gap   : {mean_mcc_train - mean_mcc_test:.4f}")
        else:
                print(f"    MCC Train-Test Gap   : N/A")

    print("="*40) # Print separator after each task block

    # 8. Save Results
    # (Call remains the same, ensure save_results definition matches)
    save_results(MODEL_FILENAME, vectorizer, ensemble_models, TASK_NAMES,
                 best_params,
                 mean_cv_test_scores_auc, mean_cv_train_scores_auc,
                 mean_cv_test_scores_f1, mean_cv_train_scores_f1,
                 mean_cv_test_scores_bal_acc, mean_cv_train_scores_bal_acc,
                 mean_cv_test_scores_mcc, mean_cv_train_scores_mcc,
                 optimal_thresholds)

    # 9. Post-Training Analysis (Optional)
    # --- MODIFIED: run_post_training_analysis needs internal changes for predict_proba ---
    # The call signature remains the same (passing optimal_thresholds)
    if X_vec is not None and Y is not None:
         post_figs = run_post_training_analysis(ensemble_models, vectorizer, X_vec, Y, TASK_NAMES,
                                                optimal_thresholds, plots_dir=PLOTS_DIR)
         #all_figures_to_save.update(post_figs) # Add figures if returned
    # --- END MODIFICATION ---


    # 10. Final Plot Saving Loop (if functions return figures)
    print("\n--- Saving All Generated Plots ---")
    if not all_figures_to_save:
        print("No figures were generated/collected to save.")
    else:
        # ... (The existing loop to save figures from all_figures_to_save) ...
        pass # Add saving loop here if needed

    end_time = time.time()
    print(f"\n🚀 Total process completed in {end_time - start_time:.2f} seconds.")
    print("Main script finished.")
    plt.show()