# === Imports ===
import numpy as np
import os
import pickle
import warnings # To catch convergence warnings
from typing import List, Dict, Tuple, Optional, Any

from sklearn.base import clone, BaseEstimator
from sklearn.exceptions import ConvergenceWarning # To specifically catch convergence warnings
from sklearn.metrics import ( # Import all necessary metrics
    roc_auc_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.utils import resample
# Assuming MLPClassifier is used in the pipeline passed via best_estimators
# No direct import needed here if it's inside the pipeline object

# === Function Definition ===

def train_evaluate_bootstrap(
    best_estimators: Dict[str, Optional[BaseEstimator]], # Dict of best pipelines found per task
    X_vec: np.ndarray,
    Y: np.ndarray,
    task_names: List[str],
    n_splits: int,
    n_bootstrap: int,
    random_state: int
    ) -> Tuple[
        Dict[str, List[BaseEstimator]], # ensemble_models
        # Mean CV Scores (Test & Train for each metric)
        Dict[str, float], Dict[str, float], # AUC
        Dict[str, float], Dict[str, float], # F1
        Dict[str, float], Dict[str, float], # Balanced Accuracy
        Dict[str, float], Dict[str, float], # MCC
        # OOF Predictions (Probabilities)
        Dict[str, Dict[str, np.ndarray]]
    ]:

    print(f"\n--- K-Fold CV Training ({n_splits} folds) & Bootstrapping ({n_bootstrap} iterations) ---")

    # --- Initialize storage ---
    ensemble_models: Dict[str, List[BaseEstimator]] = {name: [] for name in task_names}
    # Store lists of scores per fold for each metric
    fold_test_scores_auc: Dict[str, List[float]] = {name: [] for name in task_names}
    fold_train_scores_auc: Dict[str, List[float]] = {name: [] for name in task_names}
    fold_test_scores_f1: Dict[str, List[float]] = {name: [] for name in task_names}
    fold_train_scores_f1: Dict[str, List[float]] = {name: [] for name in task_names}
    fold_test_scores_bal_acc: Dict[str, List[float]] = {name: [] for name in task_names}
    fold_train_scores_bal_acc: Dict[str, List[float]] = {name: [] for name in task_names}
    fold_test_scores_mcc: Dict[str, List[float]] = {name: [] for name in task_names}
    fold_train_scores_mcc: Dict[str, List[float]] = {name: [] for name in task_names}
    # Store final mean scores
    final_mean_test_scores_auc: Dict[str, float] = {name: np.nan for name in task_names}
    final_mean_train_scores_auc: Dict[str, float] = {name: np.nan for name in task_names}
    final_mean_test_scores_f1: Dict[str, float] = {name: np.nan for name in task_names}
    final_mean_train_scores_f1: Dict[str, float] = {name: np.nan for name in task_names}
    final_mean_test_scores_bal_acc: Dict[str, float] = {name: np.nan for name in task_names}
    final_mean_train_scores_bal_acc: Dict[str, float] = {name: np.nan for name in task_names}
    final_mean_test_scores_mcc: Dict[str, float] = {name: np.nan for name in task_names}
    final_mean_train_scores_mcc: Dict[str, float] = {name: np.nan for name in task_names}
    # Store out-of-fold (OOF) predictions (probabilities) and labels
    oof_predictions_per_task: Dict[str, Dict[str, List]] = {
        name: {'scores': [], 'labels': []} for name in task_names
    }
    # Counters for convergence warnings
    total_cv_convergence_warnings = 0
    total_boot_convergence_warnings = 0
    # --- End Initialization ---


    # --- K-Fold Loop ---
    mlskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(mlskf.split(X_vec, Y)):
        print(f"\n📦 Fold {fold + 1}/{n_splits}")
        X_train_fold, Y_train_fold = X_vec[train_idx], Y[train_idx]
        X_test_fold, Y_test_fold = X_vec[test_idx], Y[test_idx]

        # --- Task Loop ---
        for task_index, task_name in enumerate(task_names):
            print(f"  Processing Task: {task_name}")
            best_pipeline_config = best_estimators.get(task_name)
            y_train_task_fold = Y_train_fold[:, task_index]
            y_test_task_fold = Y_test_fold[:, task_index]

            # Skip if tuning failed for this task
            if best_pipeline_config is None:
                print(f"    Skipping Task in Fold {fold+1}: No best estimator from tuning.")
                # Append NaNs to all score lists for this fold/task
                for score_list in [fold_test_scores_auc, fold_train_scores_auc,
                                   fold_test_scores_f1, fold_train_scores_f1,
                                   fold_test_scores_bal_acc, fold_train_scores_bal_acc,
                                   fold_test_scores_mcc, fold_train_scores_mcc]:
                    score_list[task_name].append(np.nan)
                continue # Skip CV scoring and bootstrapping

            # --- CV Scoring (using predict_proba) ---
            print(f"    Training/evaluating single model for CV score...")
            cv_pipeline = clone(best_pipeline_config)
            # Initialize scores for this fold/task
            current_fold_test_score_auc = np.nan
            current_fold_train_score_auc = np.nan
            current_fold_test_score_f1 = np.nan
            current_fold_train_score_f1 = np.nan
            current_fold_test_score_bal_acc = np.nan
            current_fold_train_score_bal_acc = np.nan
            current_fold_test_score_mcc = np.nan
            current_fold_train_score_mcc = np.nan

            try:
                # Check for single class issues
                unique_train_labels = np.unique(y_train_task_fold)
                unique_test_labels = np.unique(y_test_task_fold)

                if len(unique_train_labels) < 2:
                     print(f"    ⚠️ Skipping CV scores: Only one class ({unique_train_labels[0]}) in train data.")
                elif len(unique_test_labels) < 2:
                     print(f"    ⚠️ Skipping Test CV scores: Only one class ({unique_test_labels[0]}) in test data.")
                     # Try to fit and get train scores anyway
                     try:
                         # --- Catch Warnings during CV fit (Train Only) ---
                         with warnings.catch_warnings(record=True) as w_cv_trainonly:
                             warnings.simplefilter("always", ConvergenceWarning)
                             cv_pipeline.fit(X_train_fold, y_train_task_fold)
                             for warn_cv in w_cv_trainonly:
                                 if issubclass(warn_cv.category, ConvergenceWarning):
                                     print(f"    ⚠️ ConvergenceWarning during CV fit for {task_name} (Fold {fold+1}, Train Only).")
                                     total_cv_convergence_warnings += 1
                         # --- End Catch Warnings ---

                         # Train Scores using predict_proba
                         y_proba_train = cv_pipeline.predict_proba(X_train_fold)[:, 1]
                         y_pred_train = (y_proba_train > 0.5).astype(int) # Threshold at 0.5

                         current_fold_train_score_auc = roc_auc_score(y_train_task_fold, y_proba_train)
                         current_fold_train_score_f1 = f1_score(y_train_task_fold, y_pred_train, zero_division=0)
                         current_fold_train_score_bal_acc = balanced_accuracy_score(y_train_task_fold, y_pred_train)
                         current_fold_train_score_mcc = matthews_corrcoef(y_train_task_fold, y_pred_train)
                         print(f"    📈 Train AUC:{current_fold_train_score_auc:.4f}, F1 :{current_fold_train_score_f1:.4f}, BalAcc:{current_fold_train_score_bal_acc:.4f}, MCC:{current_fold_train_score_mcc:.4f} (Test scores skipped)")
                     except Exception as e_train:
                         print(f"    ⚠️ Error calculating Train Scores (Test skipped): {e_train}")
                         # Ensure train scores remain NaN
                         current_fold_train_score_auc = np.nan
                         current_fold_train_score_f1 = np.nan
                         current_fold_train_score_bal_acc = np.nan
                         current_fold_train_score_mcc = np.nan
                else:
                     # --- Fit CV model and Catch Warnings ---
                     with warnings.catch_warnings(record=True) as w_cv:
                         warnings.simplefilter("always", ConvergenceWarning)
                         print(f"    Fitting CV model for {task_name}...")
                         cv_pipeline.fit(X_train_fold, y_train_task_fold)
                         for warn_cv in w_cv:
                             if issubclass(warn_cv.category, ConvergenceWarning):
                                 print(f"    ⚠️ ConvergenceWarning during CV fit for {task_name} (Fold {fold+1}).")
                                 total_cv_convergence_warnings += 1
                     # --- End Catch Warnings ---

                     # --- Test Scores using predict_proba ---
                     y_proba_test = cv_pipeline.predict_proba(X_test_fold)[:, 1] # Get probabilities
                     y_pred_test = (y_proba_test > 0.5).astype(int) # Threshold at 0.5

                     current_fold_test_score_auc = roc_auc_score(y_test_task_fold, y_proba_test)
                     current_fold_test_score_f1 = f1_score(y_test_task_fold, y_pred_test, zero_division=0)
                     current_fold_test_score_bal_acc = balanced_accuracy_score(y_test_task_fold, y_pred_test)
                     current_fold_test_score_mcc = matthews_corrcoef(y_test_task_fold, y_pred_test)
                     print(f"    📊 Test AUC :{current_fold_test_score_auc:.4f}, F1 :{current_fold_test_score_f1:.4f}, BalAcc:{current_fold_test_score_bal_acc:.4f}, MCC:{current_fold_test_score_mcc:.4f}")

                     # --- Store OOF probabilities and labels ---
                     oof_predictions_per_task[task_name]['scores'].extend(y_proba_test.tolist()) # Store probabilities
                     oof_predictions_per_task[task_name]['labels'].extend(y_test_task_fold.tolist())
                     # --- End Store OOF ---

                     # --- Train Scores using predict_proba ---
                     try:
                         y_proba_train = cv_pipeline.predict_proba(X_train_fold)[:, 1] # Get probabilities
                         y_pred_train = (y_proba_train > 0.5).astype(int) # Threshold at 0.5

                         current_fold_train_score_auc = roc_auc_score(y_train_task_fold, y_proba_train)
                         current_fold_train_score_f1 = f1_score(y_train_task_fold, y_pred_train, zero_division=0)
                         current_fold_train_score_bal_acc = balanced_accuracy_score(y_train_task_fold, y_pred_train)
                         current_fold_train_score_mcc = matthews_corrcoef(y_train_task_fold, y_pred_train)
                         print(f"    📈 Train AUC:{current_fold_train_score_auc:.4f}, F1 :{current_fold_train_score_f1:.4f}, BalAcc:{current_fold_train_score_bal_acc:.4f}, MCC:{current_fold_train_score_mcc:.4f}")
                     except Exception as e_train:
                         print(f"    ⚠️ Error calculating Train Scores: {e_train}")
                         # Set train scores to NaN if calculation fails
                         current_fold_train_score_auc = np.nan
                         current_fold_train_score_f1 = np.nan
                         current_fold_train_score_bal_acc = np.nan
                         current_fold_train_score_mcc = np.nan

            except Exception as e:
                 print(f"    ⚠️ Unexpected Error during CV scoring phase for {task_name}: {e}")
                 # Ensure NaNs are appended if a major error occurs before scoring
                 current_fold_test_score_auc = np.nan
                 current_fold_train_score_auc = np.nan
                 current_fold_test_score_f1 = np.nan
                 current_fold_train_score_f1 = np.nan
                 current_fold_test_score_bal_acc = np.nan
                 current_fold_train_score_bal_acc = np.nan
                 current_fold_test_score_mcc = np.nan
                 current_fold_train_score_mcc = np.nan

            # Append all calculated scores (or NaNs) for this fold/task
            fold_test_scores_auc[task_name].append(current_fold_test_score_auc)
            fold_train_scores_auc[task_name].append(current_fold_train_score_auc)
            fold_test_scores_f1[task_name].append(current_fold_test_score_f1)
            fold_train_scores_f1[task_name].append(current_fold_train_score_f1)
            fold_test_scores_bal_acc[task_name].append(current_fold_test_score_bal_acc)
            fold_train_scores_bal_acc[task_name].append(current_fold_train_score_bal_acc)
            fold_test_scores_mcc[task_name].append(current_fold_test_score_mcc)
            fold_train_scores_mcc[task_name].append(current_fold_train_score_mcc)
            # --- End CV Scoring ---


            # --- Bootstrapping ---
            print(f"    Bootstrapping ({n_bootstrap} iterations)...")
            if len(X_train_fold) == 0:
                 print("      ⚠️ Skipping bootstrapping: No training data in fold.")
                 continue # Skip bootstrapping if fold is empty

            bootstrap_models_added_this_fold = 0
            for b in range(n_bootstrap):
                if len(X_train_fold) < 1: break # Safety break

                X_boot, y_boot_task = resample(X_train_fold, y_train_task_fold,
                                               replace=True, random_state=random_state + fold + b)

                # Debug print for bootstrap sample
                unique_labels, counts = np.unique(y_boot_task, return_counts=True)
                # print(f"      Bootstrap sample {b+1}: X_boot shape {X_boot.shape}, y_boot unique values & counts {dict(zip(unique_labels, counts))}") # Optional verbose debug

                boot_pipeline = clone(best_pipeline_config)
                try:
                    if len(unique_labels) < 2:
                         # print(f"      Skipping bootstrap sample {b+1}: Only one class present.") # Optional verbose debug
                         continue

                    # --- Catch Warnings during Bootstrap fit ---
                    with warnings.catch_warnings(record=True) as w_boot:
                        warnings.simplefilter("always", ConvergenceWarning)
                        # print(f"      Attempting to fit bootstrap sample {b+1}...") # Optional verbose debug
                        boot_pipeline.fit(X_boot, y_boot_task)
                        # Check warnings
                        for warn_boot in w_boot:
                             if issubclass(warn_boot.category, ConvergenceWarning):
                                 print(f"    ⚠️ ConvergenceWarning during Bootstrap fit {b+1} for {task_name} (Fold {fold+1}).")
                                 total_boot_convergence_warnings += 1
                    # --- End Catch Warnings ---

                    ensemble_models[task_name].append(boot_pipeline)
                    bootstrap_models_added_this_fold += 1

                except Exception as e:
                     print(f"      ⚠️ Error fitting bootstrap sample {b+1} for task {task_name}: {e}")

            print(f"      Finished bootstrapping for task {task_name} in fold {fold+1}. Added {bootstrap_models_added_this_fold}/{n_bootstrap} models.")
            # --- End Bootstrapping ---
        # --- End Task Loop ---
    # --- End Fold Loop ---


    # --- Report Total Convergence Warnings ---
    print("\n--- Convergence Summary ---")
    if total_cv_convergence_warnings > 0:
        print(f"⚠️ Total ConvergenceWarnings during CV scoring fits: {total_cv_convergence_warnings}")
    if total_boot_convergence_warnings > 0:
        print(f"⚠️ Total ConvergenceWarnings during Bootstrap fits: {total_boot_convergence_warnings}")
    if total_cv_convergence_warnings == 0 and total_boot_convergence_warnings == 0:
        print("   No ConvergenceWarnings encountered during training.")
    else:
        print("   If warnings persist, consider increasing max_iter or adjusting MLP parameters.")
    # --- End Report ---


    # --- Calculate Final Mean Scores ---
    print("\nCalculating final mean scores across folds...")
    # Helper function to calculate mean safely, ignoring NaNs
    def _safe_mean(scores_list):
        valid_scores = [s for s in scores_list if not np.isnan(s)]
        return np.mean(valid_scores) if valid_scores else np.nan

    for task_name in task_names:
        final_mean_test_scores_auc[task_name] = _safe_mean(fold_test_scores_auc.get(task_name, []))
        final_mean_train_scores_auc[task_name] = _safe_mean(fold_train_scores_auc.get(task_name, []))
        final_mean_test_scores_f1[task_name] = _safe_mean(fold_test_scores_f1.get(task_name, []))
        final_mean_train_scores_f1[task_name] = _safe_mean(fold_train_scores_f1.get(task_name, []))
        final_mean_test_scores_bal_acc[task_name] = _safe_mean(fold_test_scores_bal_acc.get(task_name, []))
        final_mean_train_scores_bal_acc[task_name] = _safe_mean(fold_train_scores_bal_acc.get(task_name, []))
        final_mean_test_scores_mcc[task_name] = _safe_mean(fold_test_scores_mcc.get(task_name, []))
        final_mean_train_scores_mcc[task_name] = _safe_mean(fold_train_scores_mcc.get(task_name, []))

    # --- Convert collected OOF lists to numpy arrays ---
    print("Converting OOF predictions to numpy arrays...")
    for task_name in task_names:
         if task_name in oof_predictions_per_task: # Check if key exists
             oof_predictions_per_task[task_name]['scores'] = np.array(oof_predictions_per_task[task_name]['scores'])
             oof_predictions_per_task[task_name]['labels'] = np.array(oof_predictions_per_task[task_name]['labels'])
         else: # Should not happen if initialized correctly, but safe check
             oof_predictions_per_task[task_name] = {'scores': np.array([]), 'labels': np.array([])}


    # --- Return all results ---
    return (ensemble_models,
            final_mean_test_scores_auc, final_mean_train_scores_auc,
            final_mean_test_scores_f1, final_mean_train_scores_f1,
            final_mean_test_scores_bal_acc, final_mean_train_scores_bal_acc,
            final_mean_test_scores_mcc, final_mean_train_scores_mcc,
            oof_predictions_per_task
            )