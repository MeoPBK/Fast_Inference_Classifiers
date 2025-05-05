# === Imports ===
import numpy as np
import matplotlib.pyplot as plt # Optional: If adding plotting inside
import os # Optional: If adding saving plots inside
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import precision_recall_curve, f1_score # Ensure imported

# === Function Definition ===

def find_optimal_thresholds(
    oof_predictions: Dict[str, Dict[str, np.ndarray]],
    task_names: List[str],
    default_threshold: float = 0.5, # <-- Default changed to 0.5 for probabilities
    create_plots: bool = False, # <-- Optional: Flag to generate P-R plots
    plots_dir: Optional[str] = None # <-- Optional: Directory to save plots
    ) -> Dict[str, float]: # Returns only the thresholds dictionary
    """
    Finds the optimal decision threshold for each task by maximizing F1-score
    on out-of-fold predicted probabilities using the Precision-Recall curve.

    Args:
        oof_predictions: Dictionary containing pooled OOF probabilities and true labels per task.
                         Format: {task: {'scores': array (probabilities 0-1), 'labels': array}}
        task_names: List of task names.
        default_threshold: Threshold to return if optimization fails for a task (0.5 is standard for probabilities).
        create_plots: If True, generate and save/show Precision-Recall curve plots for each task.
        plots_dir: Directory to save plots if create_plots is True. If None, plots are shown non-blockingly.

    Returns:
        Dictionary mapping task names to their optimal probability threshold.
    """
    print("\n--- Finding Optimal Thresholds via P-R Curve (Maximizing F1 on OOF Probs) ---")
    optimal_thresholds: Dict[str, float] = {}
    # Optional: Store figures if create_plots is True and you want to return them
    # threshold_figures: Dict[str, plt.Figure] = {}

    for task_name in task_names:
        print(f"  Processing Task: {task_name}")
        task_data = oof_predictions.get(task_name)
        threshold_found = False
        fig_pr = None # Initialize figure variable for potential plot

        if task_data and len(task_data['scores']) > 0 and len(task_data['labels']) > 0:
            y_true = task_data['labels']
            y_scores = task_data['scores'] # These are now probabilities (0-1)

            # Check if there are at least two classes in the true labels collected
            if len(np.unique(y_true)) < 2:
                print("    ⚠️ Skipping threshold optimization: Only one class found in OOF labels.")
            else:
                try:
                    # Calculate precision, recall, and corresponding probability thresholds
                    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

                    # Calculate F1 score for each threshold point (thresholds array is one element shorter)
                    # Handle potential division by zero where precision+recall is 0
                    f1_scores = np.zeros_like(thresholds) # Initialize F1 scores array
                    # Avoid division by zero: only calculate F1 where precision or recall is non-zero
                    # Use precision[:-1] and recall[:-1] to match the length of thresholds
                    denominator = precision[:-1] + recall[:-1]
                    mask = denominator > 0 # Create a mask for valid calculations
                    f1_scores[mask] = (2 * precision[:-1][mask] * recall[:-1][mask]) / denominator[mask]

                    # Find the threshold that yields the maximum F1 score
                    if len(f1_scores) > 0:
                        best_f1_idx = np.argmax(f1_scores)
                        # The optimal threshold is the one corresponding to the best F1 score
                        optimal_thresholds[task_name] = thresholds[best_f1_idx]
                        max_f1 = f1_scores[best_f1_idx]
                        print(f"    Optimal Threshold: {optimal_thresholds[task_name]:.4f} (Max F1={max_f1:.4f})")
                        threshold_found = True

                        # --- Optional Plotting ---
                        if create_plots:
                            try:
                                fig_pr, ax1 = plt.subplots(figsize=(7, 5))
                                # Plot P-R curve
                                ax1.plot(recall[:-1], precision[:-1], label='Precision-Recall Curve', color='blue')
                                ax1.set_xlabel('Recall')
                                ax1.set_ylabel('Precision', color='blue')
                                ax1.tick_params(axis='y', labelcolor='blue')
                                ax1.set_ylim(0.0, 1.05)
                                ax1.set_xlim(0.0, 1.0)
                                ax1.grid(True, linestyle='--', alpha=0.6)

                                # Plot F1 vs Threshold on a secondary axis
                                ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
                                ax2.plot(recall[:-1], f1_scores, label='F1 Score vs Recall', color='red', linestyle='--')
                                ax2.set_ylabel('F1 Score', color='red')
                                ax2.tick_params(axis='y', labelcolor='red')
                                ax2.set_ylim(0.0, 1.05)

                                # Mark the chosen threshold/point
                                best_recall = recall[best_f1_idx]
                                best_precision = precision[best_f1_idx]
                                ax1.scatter(best_recall, best_precision, marker='o', color='black', s=50, zorder=5,
                                            label=f'Max F1 Point (Thresh={optimal_thresholds[task_name]:.3f})')

                                fig_pr.suptitle(f'Precision-Recall & F1 Curve: {task_name}')
                                fig_pr.legend(loc='lower left') # Combine legends (might need adjustment)
                                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                                if plots_dir:
                                    os.makedirs(plots_dir, exist_ok=True)
                                    safe_task_name = "".join(c if c.isalnum() else "_" for c in task_name)
                                    filename = os.path.join(plots_dir, f"threshold_pr_curve_{safe_task_name}.png")
                                    fig_pr.savefig(filename, dpi=150, bbox_inches='tight')
                                    print(f"    Saved P-R curve plot to: {filename}")
                                    plt.close(fig_pr)
                                else:
                                    plt.pause(0.1) # Show non-blockingly

                            except Exception as plot_e:
                                print(f"    ⚠️ Error generating P-R curve plot: {plot_e}")
                                if fig_pr: plt.close(fig_pr)
                        # --- End Optional Plotting ---

                    else:
                         print("    ⚠️ Skipping threshold optimization: No valid F1 scores calculated (precision/recall might be zero).")

                except ValueError as ve:
                     print(f"    ⚠️ Skipping threshold optimization: Error during P-R curve calculation ({ve}). Check if scores are constant or invalid.")
                except Exception as e:
                     print(f"    ⚠️ Skipping threshold optimization: Unexpected error ({e}).")
        else:
            print("    ⚠️ Skipping threshold optimization: No OOF scores/labels collected for this task.")

        # Assign default if optimization failed
        if not threshold_found:
            optimal_thresholds[task_name] = default_threshold
            print(f"    Using default threshold: {default_threshold}")

    print("-" * 40)
    # If returning plots: return optimal_thresholds, threshold_figures
    return optimal_thresholds # Return only the thresholds dictionary