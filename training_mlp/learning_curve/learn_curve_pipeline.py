import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import learning_curve.tools.utils as ut
import seaborn as sns
from typing import List, Tuple, Optional # Import necessary types
import logging

# --- Scikit-learn Imports ---
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, learning_curve, KFold # Added KFold
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import make_scorer, hamming_loss, jaccard_score, f1_score # Added multi-label metrics
from sklearn.multioutput import MultiOutputClassifier # Added for multi-label curve

# --- Kneed Import ---
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
    print("Warning: 'kneed' library not found. Elbow detection will use max training size.")
    print("Install with: pip install kneed")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===========================================
# PER-LABEL LEARNING CURVE (Enhanced)
# ===========================================

def learn_curve_per_label(X_vec, Y, task_names, base_estimator_pipeline, # Pass the base pipeline structure
                         n_splits_lc=5, random_state_lc=42, train_sizes=np.linspace(0.25, 1.0, 8)
                         )-> Tuple[np.ndarray, List[np.ndarray], List[Optional[float]]]:
    """Generates learning curves for each label using ROC AUC."""
    n_labels = Y.shape[1]
    n_points = len(train_sizes)

    all_train_curves = []
    all_test_curves = []
    elbow_samples = []
    actual_train_sizes_abs = None

    plt.figure(figsize=(16, 5 * n_labels)) # Adjusted figsize slightly

    lc_cv = StratifiedKFold(n_splits=n_splits_lc, shuffle=True, random_state=random_state_lc)

    for i in range(n_labels):
        task_name = task_names[i]
        print(f"\n📚 Learning curve for label {i+1}: {task_name}")

        # Clone the base estimator structure for this task's curve
        estimator_lc = clone(base_estimator_pipeline)
        
        # *** Dynamically find and simplify StackingClassifier's internal CV for LC ***
        stacker_found_and_set = False
        if hasattr(estimator_lc, 'steps'): # Check if it's a Pipeline
            for step_name, step_instance in estimator_lc.steps:
                if isinstance(step_instance, StackingClassifier):
                    try:
                        original_cv = step_instance.cv # Store original for info if needed
                        new_cv = KFold(n_splits=3, shuffle=True, random_state=random_state_lc + i)
                        step_instance.cv = new_cv
                        print(f"   Found StackingClassifier step: '{step_name}'. Temporarily setting internal CV to: {new_cv}")
                        stacker_found_and_set = True
                        break # Assume only one stacker, stop searching
                    except AttributeError:
                        print(f"   Warning: Could not set 'cv' attribute on StackingClassifier step '{step_name}'.")
                    except Exception as e:
                        print(f"   Warning: Error modifying CV for StackingClassifier step '{step_name}': {e}")
        if not stacker_found_and_set:
            # Optional: Add a check if the base_estimator_pipeline itself is a StackingClassifier
            if isinstance(estimator_lc, StackingClassifier):
                 try:
                     original_cv = estimator_lc.cv
                     new_cv = KFold(n_splits=3, shuffle=True, random_state=random_state_lc + i)
                     estimator_lc.cv = new_cv
                     print(f"   Base estimator is a StackingClassifier. Temporarily setting internal CV to: {new_cv}")
                     stacker_found_and_set = True
                 except AttributeError:
                     print(f"   Warning: Could not set 'cv' attribute on the base StackingClassifier.")
                 except Exception as e:
                     print(f"   Warning: Error modifying CV for the base StackingClassifier: {e}")

        if not stacker_found_and_set:
             print("   Note: No StackingClassifier found or modified in the pipeline for internal CV adjustment.")

        y_task = Y[:, i]
        scorer = 'roc_auc' # Use string identifier

        if len(np.unique(y_task)) < 2:
             print(f"⚠️ Skipping label {i} ({task_name}) - only one class present.")
             all_train_curves.append([np.nan] * n_points)
             all_test_curves.append([np.nan] * n_points)
             elbow_samples.append(np.nan)
             # Plot placeholder subplot? Or skip entirely? Let's skip plotting for now.
             # plt.subplot(n_labels, 1, i + 1)
             # plt.title(f"Learning Curve: {task_name} (Skipped - Single Class)")
             # plt.text(0.5, 0.5, 'Skipped - Single Class Present', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             continue

        print(f"   Calculating learning curve using {type(lc_cv).__name__}...")
        try:
            train_sizes_abs, train_scores, test_scores = learning_curve(
                estimator=estimator_lc, # Use the modified estimator with simplified internal CV
                X=X_vec,
                y=y_task,
                cv=lc_cv, # Outer CV for learning curve points
                scoring=scorer,
                train_sizes=train_sizes,
                n_jobs=-1,
                random_state=random_state_lc,
                error_score='raise' # Crucial to see the exact error
            )

            # Store the absolute sizes from the first successful run
            if actual_train_sizes_abs is None and train_sizes_abs is not None:
                 actual_train_sizes_abs = train_sizes_abs

            train_mean = np.nanmean(train_scores, axis=1)
            test_mean = np.nanmean(test_scores, axis=1)
            all_train_curves.append(train_mean)
            all_test_curves.append(test_mean)

            # --- Standard Deviation Bands ---
            train_std = np.nanstd(train_scores, axis=1)
            test_std = np.nanstd(test_scores, axis=1)

            # --- Smoothing ---
            test_smoothed = np.full_like(test_mean, np.nan)
            valid_test_points = ~np.isnan(test_mean)
            if np.sum(valid_test_points) >= 5:
                 test_smoothed[valid_test_points] = savgol_filter(test_mean[valid_test_points], 5, 2)
            elif np.sum(valid_test_points) > 0:
                 test_smoothed[valid_test_points] = test_mean[valid_test_points]

            # --- Elbow Detection (using Kneed) ---
            y_data_for_elbow = test_smoothed if not np.all(np.isnan(test_smoothed)) else test_mean
            elbow_x = ut.detect_elbow_kneed(train_sizes_abs, y_data_for_elbow, verbose=True)
            elbow_samples.append(elbow_x)

            # --- Plotting ---
            plt.subplot(n_labels, 1, i + 1)
            # Plot means
            plt.plot(train_sizes_abs, train_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes_abs, test_mean, 'o-', color="g", label="Cross-validation score", alpha=0.7)

            # Plot smoothed curve if available
            if not np.all(np.isnan(test_smoothed)):
                 plt.plot(train_sizes_abs, test_smoothed, '--', color="darkorange", label="CV score (smoothed)", linewidth=2)

            # Plot standard deviation bands
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
            plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)

            # Plot elbow line
            if not np.isnan(elbow_x) and elbow_x > 0:
                 plt.axvline(elbow_x, color='blue', linestyle='--', label=f'Elbow (~{elbow_x} samples)')

            plt.title(f"Learning Curve: {task_name} (ROC AUC)")
            plt.xlabel("Training examples")
            plt.ylabel("ROC AUC")
            plt.grid(True)
            plt.ylim(0.9, 1.05)
            plt.legend(loc="best")

        except Exception as e: # Catch broader errors during LC calculation
            print(f"⚠️ Error computing learning curve for label {i} ({task_name}): {e}")
            all_train_curves.append([np.nan] * n_points)
            all_test_curves.append([np.nan] * n_points)
            elbow_samples.append(np.nan)
            # Optionally plot a placeholder indicating error
            traceback.print_exc()
            plt.subplot(n_labels, 1, i + 1)
            plt.title(f"Learning Curve: {task_name} (Error)")
            plt.text(0.5, 0.5, f'Error during calculation:\n{e}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, wrap=True)
            continue

    plt.suptitle("Per-Label Learning Curves (ROC AUC)", fontsize=16, y=1.02) # Add overall title
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout
    plt.show(block=False)
    plt.pause(5) # Pause to allow viewing before closing

    # --- Combined plot ---
    plt.figure(figsize=(10, 6))
    # Determine the x-axis for the combined plot
    plot_x_axis = None
    x_label = "" # Initialize x-axis label
    if actual_train_sizes_abs is not None:
        # Use the actual absolute sizes if they were captured
        plot_x_axis = actual_train_sizes_abs
        x_label = "Training examples"
        print("\n📊 Generating Combined Plot using actual training sizes.")
    else:
        # Fallback to approximation if no learning curve succeeded
        # Use X_vec.shape[0] which is standard for getting sample count from numpy array
        plot_x_axis = train_sizes * X_vec.shape[0]
        x_label = "Training examples (Approximate)"
        print("\n📊 Generating Combined Plot using approximate training sizes (fallback).")

    # Only proceed with plotting if we determined a valid x-axis
    if plot_x_axis is not None:
        # The loop for plotting each curve starts here
        for i in range(len(all_test_curves)):
             if not np.all(np.isnan(all_test_curves[i])):
                 valid_points = ~np.isnan(all_test_curves[i])
                 current_curve = all_test_curves[i]
                 # Ensure x-axis length matches curve length (handles potential partial failures)
                 current_x = plot_x_axis[:len(current_curve)]

                 if np.sum(valid_points) >= 5:
                     smoothed = np.full_like(current_curve, np.nan)
                     smoothed[valid_points] = savgol_filter(current_curve[valid_points], 5, 2)
                 else:
                     smoothed = current_curve
                 # Use the determined x-axis variable 'current_x' here
                 plt.plot(current_x, smoothed, label=f"{task_names[i]}")

        plt.title("📊 Combined Smoothed Validation Curves (ROC AUC)")
        plt.xlabel(x_label) # Use the determined x-axis label
        plt.ylabel("Validation ROC AUC")
        # plt.ylim(0.9, 1.05) # Consider making this dynamic too
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(5)
    else:
         # If actual_train_sizes_abs was None (meaning all per-label curves failed)
         print("   Skipping combined plot as no learning curves were successfully generated.")
    return train_sizes, all_test_curves, elbow_samples


# ===========================================
# MULTI-LABEL LEARNING CURVE (New)
# ===========================================

def plot_multilabel_learning_curve(estimator, X, Y, cv_splits=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring_metric='jaccard_samples', random_state=42):
    """Plots a learning curve using a multi-label performance metric."""

    # 1. Clone the input estimator to avoid modifying the original
    estimator_lc = clone(estimator)

    # 2. Dynamically find and simplify StackingClassifier's internal CV for this LC plot
    stacker_found_and_set_ml = False
    if hasattr(estimator_lc, 'steps'): # Check if it's a Pipeline
        for step_name, step_instance in estimator_lc.steps:
            if isinstance(step_instance, StackingClassifier):
                try:
                    original_cv_ml = step_instance.cv
                    new_cv_ml = KFold(n_splits=3, shuffle=True, random_state=random_state) # Use the function's random_state
                    step_instance.cv = new_cv_ml
                    print(f"   Found StackingClassifier step: '{step_name}'. Temporarily setting internal CV to: {new_cv_ml} for multi-label LC plot.")
                    stacker_found_and_set_ml = True
                    break
                except AttributeError:
                    print(f"   Warning: Could not set 'cv' attribute on StackingClassifier step '{step_name}' for multi-label LC plot.")
                except Exception as e:
                    print(f"   Warning: Error modifying CV for StackingClassifier step '{step_name}' for multi-label LC plot: {e}")

    if not stacker_found_and_set_ml:
         # Optional: Add a check if the base_estimator_pipeline itself is a StackingClassifier
         if isinstance(estimator_lc, StackingClassifier):
             try:
                 original_cv_ml = estimator_lc.cv
                 new_cv_ml = KFold(n_splits=3, shuffle=True, random_state=random_state)
                 estimator_lc.cv = new_cv_ml
                 print(f"   Base estimator is a StackingClassifier. Temporarily setting internal CV to: {new_cv_ml} for multi-label LC plot.")
                 stacker_found_and_set_ml = True
             except AttributeError:
                 print(f"   Warning: Could not set 'cv' attribute on the base StackingClassifier for multi-label LC plot.")
             except Exception as e:
                 print(f"   Warning: Error modifying CV for the base StackingClassifier for multi-label LC plot: {e}")

    if not stacker_found_and_set_ml:
         print("   Note: No StackingClassifier found or modified in the pipeline for internal CV adjustment in multi-label LC plot.")

    # 3. Wrap the estimator for multi-output handling
    multi_output_estimator = MultiOutputClassifier(estimator, n_jobs=-1)

    # Define the scorer
    scorer = None
    greater_is_better = True
    if scoring_metric == 'hamming_loss':
        scorer = make_scorer(hamming_loss, greater_is_better=False)
        greater_is_better = False # Lower is better
    elif scoring_metric == 'jaccard_samples':
        scorer = make_scorer(jaccard_score, average='samples', zero_division=1)    
    elif scoring_metric == 'f1_micro':
         scorer = make_scorer(f1_score, average='micro')
    elif scoring_metric == 'f1_macro':
         scorer = make_scorer(f1_score, average='macro')
    elif scoring_metric == 'f1_samples':
         scorer = make_scorer(f1_score, average='samples')
    else:
        try:
            # Check if it's a built-in scorer string sklearn knows
            from sklearn.metrics import get_scorer
            scorer = get_scorer(scoring_metric)
            # Try to infer greater_is_better, default to True
            greater_is_better = getattr(scorer, '_sign', 1) == 1
        except ValueError:
             print(f"Error: Unknown or unsupported scoring_metric '{scoring_metric}'")
             return

    # Use KFold for multi-label Y as StratifiedKFold/MultilabelStratifiedKFold aren't directly compatible here
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    print(f"\n📊 Generating Multi-Label Learning Curve ({scoring_metric})...")
    print(f"   Using {type(cv).__name__} for cross-validation.")

    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            multi_output_estimator, X, Y, cv=cv, scoring=scorer,
            train_sizes=train_sizes, n_jobs=-1, error_score=np.nan 
        )
        # Check if scores contain NaNs, indicating errors occurred
        if np.isnan(train_scores).any() or np.isnan(test_scores).any():
            print("   WARNING: NaNs detected in scores. Some folds failed during learning curve calculation.")
    except ValueError as e:
        # This might still catch errors before individual folds run
        print(f"Error during learning_curve setup with multi-label metric: {e}")
        import traceback
        traceback.print_exc()
        return
    except Exception as e:
        print(f"Unexpected error during learning_curve with multi-label metric: {e}")
        import traceback
        traceback.print_exc()
        return

    # Adjust scores if lower is better (like hamming_loss)
    if not greater_is_better:
        train_scores = -train_scores
        test_scores = -test_scores

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.title(f"Multi-Label Learning Curve ({scoring_metric})")
    plt.xlabel("Training examples")
    plt.ylabel(f"Score ({scoring_metric}) {' (Higher is Better)' if greater_is_better else ' (Lower is Better)'}")
    plt.grid()

    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

# ===========================================
# LABEL CO-OCCURRENCE PLOT (New)
# ===========================================

def plot_label_cooccurrence(Y, task_names):
    """Plots a heatmap of label co-occurrence."""
    print("\n📊 Generating Label Co-occurrence Heatmap...")
    if isinstance(Y, np.ndarray):
        df_Y = pd.DataFrame(Y, columns=task_names)
    else: # Assuming Y is already a DataFrame
        df_Y = Y

    cooccurrence_matrix = df_Y.T.dot(df_Y)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cooccurrence_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=task_names, yticklabels=task_names)
    plt.title('Label Co-occurrence Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)