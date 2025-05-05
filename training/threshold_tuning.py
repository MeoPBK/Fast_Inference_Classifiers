from sklearn.metrics import precision_recall_curve, f1_score # Ensure imported
import numpy as np # Ensure imported
import matplotlib.pyplot as plt 
from typing import Dict, List, Tuple, Optional

# Finds the optimal decision threshold for each task by maximizing F1-score 
# on out-of-fold predictions using the Precision-Recall curve.
def find_optimal_thresholds(
    oof_predictions: Dict[str, Dict[str, np.ndarray]],
    task_names: List[str],
    default_threshold: float = 0.0, # Default threshold if calculation fails
    create_plots: bool = False # <-- Add flag to control plotting
    ) -> Tuple[Dict[str, float], Dict[str, plt.Figure]]: # <-- Return figures dict
    
    print("\n--- Finding Optimal Thresholds via P-R Curve (Maximizing F1) ---")
    optimal_thresholds: Dict[str, float] = {}
    diagnostic_plots: Dict[str, plt.Figure] = {} # <-- Initialize dict for plots

    for task_name in task_names:
        print(f"  Processing Task: {task_name}")
        task_data = oof_predictions.get(task_name)
        threshold_found = False
        fig = None # Initialize fig to None for this task

        if task_data and len(task_data['scores']) > 0 and len(task_data['labels']) > 0:
            y_true = task_data['labels']
            y_scores = task_data['scores']

            if len(np.unique(y_true)) < 2:
                print("    ⚠️ Skipping threshold optimization: Only one class found in OOF labels.")
            else:
                try:
                    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
                    # Calculate F1 scores (same logic as before)
                    f1_scores = np.zeros_like(thresholds)
                    non_zero_pr = (precision[:-1] + recall[:-1]) > 0
                    f1_scores[non_zero_pr] = (2 * precision[:-1][non_zero_pr] * recall[:-1][non_zero_pr] /
                                              (precision[:-1][non_zero_pr] + recall[:-1][non_zero_pr]))

                    if len(f1_scores) > 0:
                        best_f1_idx = np.argmax(f1_scores)
                        optimal_threshold = thresholds[best_f1_idx]
                        max_f1 = f1_scores[best_f1_idx]
                        optimal_thresholds[task_name] = optimal_threshold
                        print(f"    Optimal Threshold: {optimal_threshold:.4f} (Max F1={max_f1:.4f})")
                        threshold_found = True

                        # --- Create Plot if requested ---
                        if create_plots:
                            print("      Generating diagnostic plot...")
                            fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # Create figure with 2 subplots

                            # P-R Curve Plot
                            axes[0].plot(recall, precision, marker='.', label='Precision-Recall Curve')
                            # Mark the operating point for the chosen threshold (approximate)
                            # Find recall/precision corresponding to the best threshold
                            # Note: Thresholds align with precision/recall starting from index 1
                            idx_thresh = np.searchsorted(thresholds, optimal_threshold) # Find index closest to threshold
                            if idx_thresh < len(precision) -1: # Ensure index is valid
                                axes[0].plot(recall[idx_thresh+1], precision[idx_thresh+1], 'ro', markersize=8, label=f'Max F1 Op Point (Th={optimal_threshold:.2f})')
                            axes[0].set_xlabel("Recall (OOF)")
                            axes[0].set_ylabel("Precision (OOF)")
                            axes[0].set_title(f"{task_name} - P-R Curve")
                            axes[0].grid(True)
                            axes[0].legend()
                            axes[0].set_xlim([-0.05, 1.05])
                            axes[0].set_ylim([-0.05, 1.05])

                            # F1 vs Threshold Plot
                            axes[1].plot(thresholds, f1_scores, marker='.', label='F1 Score')
                            axes[1].plot(optimal_threshold, max_f1, 'ro', markersize=8, label=f'Max F1 (Th={optimal_threshold:.2f})')
                            axes[1].set_xlabel("Decision Threshold (OOF Scores)")
                            axes[1].set_ylabel("F1 Score")
                            axes[1].set_title(f"{task_name} - F1 vs Threshold")
                            axes[1].grid(True)
                            axes[1].legend()

                            plt.tight_layout()
                            safe_task_name = "".join(c if c.isalnum() else "_" for c in task_name)
                            diagnostic_plots[f"pr_f1_{safe_task_name}"] = fig # Store the figure
                            print("      Diagnostic plot figure created.")
                        # --- End Plot Creation ---

                    else:
                         print("    ⚠️ Skipping threshold optimization: No valid F1 scores calculated.")

                except ValueError as ve:
                     print(f"    ⚠️ Skipping threshold optimization: Error during P-R curve calculation ({ve}).")
                except Exception as e:
                     print(f"    ⚠️ Skipping threshold optimization: Unexpected error ({e}).")

                # Close figure if plotting failed after creation
                if create_plots and not threshold_found and fig is not None:
                    plt.close(fig)

        else:
            print("    ⚠️ Skipping threshold optimization: No OOF scores/labels collected.")

        # Assign default if optimization failed
        if not threshold_found:
            optimal_thresholds[task_name] = default_threshold
            print(f"    Using default threshold: {default_threshold}")
            # Ensure no figure is stored if threshold wasn't found
            safe_task_name = "".join(c if c.isalnum() else "_" for c in task_name)
            plot_key = f"pr_f1_{safe_task_name}"
            if plot_key in diagnostic_plots:
                 plt.close(diagnostic_plots[plot_key]) # Close if created but threshold failed
                 del diagnostic_plots[plot_key]

    print("-" * 40)
    # MODIFY return statement
    return optimal_thresholds