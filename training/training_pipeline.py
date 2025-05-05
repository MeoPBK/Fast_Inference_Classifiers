# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
import time
import os # Added for PLOTS_DIR handling

from scipy.stats import loguniform
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Assuming your custom modules are structured correctly and importable
try:
    # Use your actual module names here
    from save_and_load import save_results, load_and_prepare_data
    from train_bootstrap import train_evaluate_bootstrap
    from param_tuning import tune_hyperparameters
    from learning_curve.run_analysis import run_diagnostic_analysis
    from post_analysis import run_post_training_analysis
    from threshold_tuning import find_optimal_thresholds
    MODULES_LOADED = True
except ImportError as ie:
    print(f"Error importing required modules: {ie}")
    print("Please ensure all custom modules (save_and_load, train_bootstrap, etc.) are accessible.")
    MODULES_LOADED = False


# === Configuration Constants ===
TASK_NAMES = [
    "Retrieve",
    "Internet",
    "ChatMemory",
    "RunCode"
]
N_SPLITS = 5
N_BOOTSTRAP = 10
N_TUNING_ITER = 50 # Number of iterations for RandomizedSearchCV
RANDOM_STATE = 42
MODEL_FILENAME = "./Classifier_Multilabel_Light/trained_models/model_multilabel_ridge_tuned_bootstrapped.pkl"
DATA_PATH = "./Classifier_Multilabel_Light/training_data/training_data_07.csv" # Example path
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PLOTS_DIR = "plots" # Define the directory to save plots (set to None to disable saving)

# === Main Execution ===
if __name__ == "__main__":
    if not MODULES_LOADED:
        print("Exiting due to missing modules.")
        exit()

    start_time = time.time()

    # 1. Load Data
    # Assuming load_and_prepare_data takes EMBEDDING_MODEL as an argument if needed
    X_vec, Y, vectorizer = load_and_prepare_data(DATA_PATH, TASK_NAMES, RANDOM_STATE, EMBEDDING_MODEL)

    if X_vec is None or Y is None or vectorizer is None:
        print("\nExiting due to data loading/preparation errors.")
        exit()

    # 2. Define Base Pipeline
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', RidgeClassifier(class_weight='balanced', random_state=RANDOM_STATE))
    ])

    # 3. Diagnostic Analysis (Optional but Recommended)
    # Pass PLOTS_DIR to potentially save plots internally
    _ = run_diagnostic_analysis(X_vec, Y, TASK_NAMES, base_pipeline, N_SPLITS, RANDOM_STATE)

    # 4. Tune Hyperparameters
    param_dist = {'estimator__alpha': loguniform(1e-1, 1e3)}
    best_estimators, best_params, tuning_figs = tune_hyperparameters(
        base_pipeline, param_dist, X_vec, Y, TASK_NAMES, N_SPLITS, N_TUNING_ITER, RANDOM_STATE
    )

    # 5. Train, Evaluate (CV), and Bootstrap Ensemble
    # Unpack all 11 return values correctly
    (ensemble_models,
     mean_cv_test_scores_auc, mean_cv_train_scores_auc,
     mean_cv_test_scores_f1, mean_cv_train_scores_f1,
     mean_cv_test_scores_bal_acc, mean_cv_train_scores_bal_acc, # <-- Added Bal Acc
     mean_cv_test_scores_mcc, mean_cv_train_scores_mcc,       # <-- Added MCC
     oof_predictions
     ) = train_evaluate_bootstrap(best_estimators,
                                  X_vec, Y, TASK_NAMES,
                                  N_SPLITS, N_BOOTSTRAP, RANDOM_STATE
                                 )

    # 6. Find Optimal Thresholds
    # Corrected: find_optimal_thresholds only returns the dictionary
    optimal_thresholds = find_optimal_thresholds(oof_predictions, TASK_NAMES, create_plots=True)
    # If you modified find_optimal_thresholds to also return plots, adjust unpacking here

    # 7. Print final evaluation scores
    print("\n--- Final Mean CV Scores ---")
    for task_name in TASK_NAMES:
        print(f"\n  Task: {task_name}")
        # Retrieve scores safely using .get()
        mean_auc_test = mean_cv_test_scores_auc.get(task_name, np.nan)
        mean_auc_train = mean_cv_train_scores_auc.get(task_name, np.nan)
        mean_f1_test = mean_cv_test_scores_f1.get(task_name, np.nan)
        mean_f1_train = mean_cv_train_scores_f1.get(task_name, np.nan)
        mean_bal_acc_test = mean_cv_test_scores_bal_acc.get(task_name, np.nan) # <-- Get Bal Acc
        mean_bal_acc_train = mean_cv_train_scores_bal_acc.get(task_name, np.nan)# <-- Get Bal Acc
        mean_mcc_test = mean_cv_test_scores_mcc.get(task_name, np.nan)       # <-- Get MCC
        mean_mcc_train = mean_cv_train_scores_mcc.get(task_name, np.nan)      # <-- Get MCC

        print(f"    Mean Test AUC    : {mean_auc_test:.4f}" if not np.isnan(mean_auc_test) else "    Mean Test AUC    : N/A")
        print(f"    Mean Train AUC   : {mean_auc_train:.4f}" if not np.isnan(mean_auc_train) else "    Mean Train AUC   : N/A")
        if not np.isnan(mean_auc_train) and not np.isnan(mean_auc_test):
             print(f"    AUC Train-Test Gap: {mean_auc_train - mean_auc_test:.4f}")

        print(f"    Mean Test F1     : {mean_f1_test:.4f}" if not np.isnan(mean_f1_test) else "    Mean Test F1     : N/A")
        print(f"    Mean Train F1    : {mean_f1_train:.4f}" if not np.isnan(mean_f1_train) else "    Mean Train F1    : N/A")
        if not np.isnan(mean_f1_train) and not np.isnan(mean_f1_test):
             print(f"    F1 Train-Test Gap : {mean_f1_train - mean_f1_test:.4f}")

        print(f"    Mean Test Bal Acc: {mean_bal_acc_test:.4f}" if not np.isnan(mean_bal_acc_test) else "    Mean Test Bal Acc: N/A") # <-- Print Bal Acc
        print(f"    Mean Train Bal Acc:{mean_bal_acc_train:.4f}" if not np.isnan(mean_bal_acc_train) else "    Mean Train Bal Acc: N/A")# <-- Print Bal Acc
        if not np.isnan(mean_bal_acc_train) and not np.isnan(mean_bal_acc_test):
             print(f"    Bal Acc Gap       : {mean_bal_acc_train - mean_bal_acc_test:.4f}")

        print(f"    Mean Test MCC    : {mean_mcc_test:.4f}" if not np.isnan(mean_mcc_test) else "    Mean Test MCC    : N/A")       # <-- Print MCC
        print(f"    Mean Train MCC   : {mean_mcc_train:.4f}" if not np.isnan(mean_mcc_train) else "    Mean Train MCC   : N/A")      # <-- Print MCC
        if not np.isnan(mean_mcc_train) and not np.isnan(mean_mcc_test):
             print(f"    MCC Gap           : {mean_mcc_train - mean_mcc_test:.4f}")
        # --- END MODIFICATION ---
        print("-" * 25)

    # 8. Save Results
    # --- Pass all score dictionaries ---
    save_results(MODEL_FILENAME, vectorizer, ensemble_models, TASK_NAMES,
                 best_params,
                 mean_cv_test_scores_auc, mean_cv_train_scores_auc,
                 mean_cv_test_scores_f1, mean_cv_train_scores_f1,
                 mean_cv_test_scores_bal_acc, mean_cv_train_scores_bal_acc, # <-- Pass Bal Acc
                 mean_cv_test_scores_mcc, mean_cv_train_scores_mcc,       # <-- Pass MCC
                 optimal_thresholds)

    # 9. Post-Training Analysis (Optional)
    if X_vec is not None and Y is not None: # Check if data is available
         # Pass optimal_thresholds and PLOTS_DIR
         run_post_training_analysis(ensemble_models, vectorizer, X_vec, Y, TASK_NAMES,
                                    optimal_thresholds, plots_dir=None)

    end_time = time.time()
    print(f"\n🚀 Total process completed in {end_time - start_time:.2f} seconds.")

    # REMOVED: Final plt.show() - plots should be saved or displayed non-blockingly by the functions
    plt.show()
    print("Main script finished.")