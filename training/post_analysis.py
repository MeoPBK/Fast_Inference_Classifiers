import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from typing import List, Dict, Tuple, Optional, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    hamming_loss, jaccard_score
)

# Make sure optimal_thresholds is accessible in the scope where this function is called
# or modify the function to accept it as an argument if needed.
# For this example, we assume it's loaded globally or passed if necessary.
# Let's modify it to accept thresholds explicitly for better encapsulation:

def run_post_training_analysis(
    ensemble_models: Dict[str, List[BaseEstimator]],
    vectorizer,
    X_vec: np.ndarray,
    Y: np.ndarray,
    task_names: List[str],
    optimal_thresholds: Dict[str, float], # <-- ADDED thresholds argument
    plots_dir: Optional[str] = None
    ) -> None: # Function doesn't need to return figures if saving internally

     print("\n--- Post-Training Analysis ---")
     auc_distributions = {task_name: [] for task_name in task_names}
     f1_distributions = {task_name: [] for task_name in task_names}
     # Array to store the average decision score per sample per task
     all_avg_decision_scores = np.zeros((X_vec.shape[0], len(task_names)))

     if not any(ensemble_models.values()):
          print("Skipping post-training analysis: No models found in the ensemble.")
          return

     print("Calculating scores and predictions for each model in the ensemble (this may take a moment)...")
     # --- Calculate Scores and Average Decision Function ---
     for task_index, task_name in enumerate(task_names):
          print(f"  Processing task: {task_name}")
          if task_name not in ensemble_models or not ensemble_models[task_name]:
               print(f"    Skipping score distributions for {task_name} (no models found).")
               # Assign a neutral average score if no models
               all_avg_decision_scores[:, task_index] = 0.0
               continue

          y_true_task = Y[:, task_index]
          if len(np.unique(y_true_task)) < 2:
               print(f"    Skipping score distributions for {task_name} (only one class in overall Y data).")
               # Assign a neutral average score if only one true class
               all_avg_decision_scores[:, task_index] = 0.0
               continue

          task_models = ensemble_models[task_name]
          task_accumulated_scores = np.zeros(X_vec.shape[0])
          valid_model_count = 0

          for model_index, model in enumerate(task_models):
               try:
                    # Primarily expect decision_function for Ridge pipelines
                    if hasattr(model, 'decision_function'):
                         y_decision = model.decision_function(X_vec)
                         # Calculate metrics based on decision function scores
                         auc = roc_auc_score(y_true_task, y_decision)
                         auc_distributions[task_name].append(auc)
                         # Calculate F1 based on thresholding decision function at 0
                         y_pred = (y_decision > 0).astype(int)
                         f1 = f1_score(y_true_task, y_pred, zero_division=0)
                         f1_distributions[task_name].append(f1)
                         # Accumulate scores for averaging
                         task_accumulated_scores += y_decision
                         valid_model_count += 1

                    # Fallback for models with predict_proba (less likely here)
                    elif hasattr(model, 'predict_proba'):
                         y_score_proba = model.predict_proba(X_vec)[:, 1]
                         auc = roc_auc_score(y_true_task, y_score_proba)
                         auc_distributions[task_name].append(auc)
                         # Calculate F1 based on thresholding probability at 0.5
                         y_pred = (y_score_proba > 0.5).astype(int)
                         f1 = f1_score(y_true_task, y_pred, zero_division=0)
                         f1_distributions[task_name].append(f1)
                         # Cannot directly average probabilities with decision scores
                         print(f"    Warning: Model {model_index} for {task_name} uses predict_proba, cannot average its score with decision_function scores.")
                         # Decide how to handle this - maybe skip averaging? For now, we don't add to task_accumulated_scores

                    else:
                         print(f"    Model {model_index} for {task_name} lacks decision_function/predict_proba, skipping score calculation.")

               except ValueError as ve:
                    print(f"    Skipping score calculation for model {model_index} in {task_name} (ValueError: {ve}).")
               except Exception as e:
                    print(f"    Error calculating score for model {model_index} in {task_name}: {e}")

          # Calculate average decision score for the task
          if valid_model_count > 0:
               all_avg_decision_scores[:, task_index] = task_accumulated_scores / valid_model_count
          else:
               print(f"    No valid models found to calculate average decision score for {task_name}.")
               all_avg_decision_scores[:, task_index] = 0.0 # Assign neutral score

     # --- Calculate Ensemble Predictions & Multi-Label Metrics ---
     print("\n--- Overall Multi-Label Ensemble Performance (using optimal thresholds) ---")
     Y_pred_ensemble = np.zeros_like(Y)
     for task_index, task_name in enumerate(task_names):
          task_threshold = optimal_thresholds.get(task_name, 0.0) # Use loaded optimal threshold
          Y_pred_ensemble[:, task_index] = (all_avg_decision_scores[:, task_index] > task_threshold).astype(int)

     try:
          hl = hamming_loss(Y, Y_pred_ensemble)
          js_samples = jaccard_score(Y, Y_pred_ensemble, average='samples', zero_division=0)
          js_macro = jaccard_score(Y, Y_pred_ensemble, average='macro', zero_division=0)
          f1_macro = f1_score(Y, Y_pred_ensemble, average='macro', zero_division=0)
          f1_micro = f1_score(Y, Y_pred_ensemble, average='micro', zero_division=0)
          print(f"  Hamming Loss     : {hl:.4f}")
          print(f"  Jaccard (Samples): {js_samples:.4f}")
          print(f"  Jaccard (Macro)  : {js_macro:.4f}")
          print(f"  F1 (Macro)       : {f1_macro:.4f}")
          print(f"  F1 (Micro)       : {f1_micro:.4f}")
     except Exception as e:
          print(f"  Error calculating multi-label metrics: {e}")
     print("-" * 40)

     # --- Plotting ---
     print("\nGenerating analysis plots...")
     tasks_with_auc_data = [name for name, dist in auc_distributions.items() if dist]
     tasks_with_f1_data = [name for name, dist in f1_distributions.items() if dist]

     # Plot AUC histograms
     if tasks_with_auc_data:
          try:
              fig_auc_hist = plt.figure(figsize=(16, 5))
              num_tasks_to_plot = len(tasks_with_auc_data)
              for i, task_name in enumerate(tasks_with_auc_data):
                   plt.subplot(1, num_tasks_to_plot, i + 1)
                   plt.hist(auc_distributions[task_name], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
                   mean_auc = np.mean(auc_distributions[task_name])
                   plt.title(f"{task_name}\nMean AUC = {mean_auc:.3f}")
                   plt.xlabel("AUC on Full Dataset")
                   plt.ylabel("Count (Models)")
                   plt.grid(axis='y', linestyle='--')
              plt.tight_layout(rect=[0, 0.03, 1, 0.95])
              plt.suptitle("AUC Distribution Across Final Ensemble Models", fontsize=14)

              if plots_dir:
                  os.makedirs(plots_dir, exist_ok=True)
                  filename = os.path.join(plots_dir, "dist_auc_histograms.png")
                  fig_auc_hist.savefig(filename, dpi=300, bbox_inches='tight')
                  print(f"  Saved AUC histogram plot to: {filename}")
                  plt.close(fig_auc_hist)
              else:
                  plt.pause(0.1)
          except Exception as e:
              print(f"  Error generating/saving AUC histogram plot: {e}")
              if 'fig_auc_hist' in locals() and fig_auc_hist is not None: plt.close(fig_auc_hist)
     else:
          print("No AUC data to plot.")

     # Plot F1 histograms
     if tasks_with_f1_data:
          try:
              fig_f1_hist = plt.figure(figsize=(16, 5))
              num_tasks_to_plot = len(tasks_with_f1_data)
              for i, task_name in enumerate(tasks_with_f1_data):
                   plt.subplot(1, num_tasks_to_plot, i + 1)
                   plt.hist(f1_distributions[task_name], bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
                   mean_f1 = np.mean(f1_distributions[task_name])
                   plt.title(f"{task_name}\nMean F1 = {mean_f1:.3f}")
                   plt.xlabel("F1 Score on Full Dataset")
                   plt.ylabel("Count (Models)")
                   plt.grid(axis='y', linestyle='--')
              plt.tight_layout(rect=[0, 0.03, 1, 0.95])
              plt.suptitle("F1 Score Distribution Across Final Ensemble Models", fontsize=14)

              if plots_dir:
                  os.makedirs(plots_dir, exist_ok=True)
                  filename = os.path.join(plots_dir, "dist_f1_histograms.png")
                  fig_f1_hist.savefig(filename, dpi=300, bbox_inches='tight')
                  print(f"  Saved F1 histogram plot to: {filename}")
                  plt.close(fig_f1_hist)
              else:
                  plt.pause(0.1)
          except Exception as e:
              print(f"  Error generating/saving F1 histogram plot: {e}")
              if 'fig_f1_hist' in locals() and fig_f1_hist is not None: plt.close(fig_f1_hist)
     else:
          print("No F1 data to plot.")

     # Plot Per-Task Confusion Matrices
     print("Generating Confusion Matrices...")
     n_tasks = len(task_names)
     ncols = min(n_tasks, 4)
     nrows = (n_tasks + ncols - 1) // ncols
     try:
         fig_cm, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4.5, nrows * 4), squeeze=False) # Adjusted size
         axes_flat = axes.flatten()

         for i, task_name in enumerate(task_names):
             if i >= len(axes_flat): break
             ax = axes_flat[i]
             y_true_task = Y[:, i]
             y_pred_task = Y_pred_ensemble[:, i] # Use ensemble predictions based on optimal thresholds

             if len(np.unique(y_true_task)) < 2:
                  ax.text(0.5, 0.5, 'Single Class\nin True Labels', ha='center', va='center', transform=ax.transAxes)
                  ax.set_title(f"{task_name}\n(Skipped)")
                  continue

             try:
                 cm = confusion_matrix(y_true_task, y_pred_task, labels=[0, 1]) # Ensure labels are 0, 1
                 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
                 disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d') # Use 'd' for integer format
                 ax.set_title(task_name)
             except Exception as e_cm:
                  ax.text(0.5, 0.5, f'Error:\n{e_cm}', ha='center', va='center', transform=ax.transAxes, wrap=True)
                  ax.set_title(f"{task_name}\n(Error)")

         # Hide any unused subplots
         for j in range(i + 1, len(axes_flat)):
             axes_flat[j].axis('off')

         plt.tight_layout(rect=[0, 0.03, 1, 0.97])
         plt.suptitle("Per-Task Confusion Matrices (Ensemble Predictions, Optimal Thresholds)", fontsize=14, y=1.0)

         if plots_dir:
             os.makedirs(plots_dir, exist_ok=True)
             filename = os.path.join(plots_dir, "confusion_matrices.png")
             fig_cm.savefig(filename, dpi=300, bbox_inches='tight')
             print(f"  Saved Confusion Matrix plot to: {filename}")
             plt.close(fig_cm)
         else:
             plt.pause(0.1)
     except Exception as e:
         print(f"  Error generating/saving Confusion Matrix plot: {e}")
         if 'fig_cm' in locals() and fig_cm is not None: plt.close(fig_cm)


     # Plot Combined Boxplot
     print("Generating combined AUC/F1 boxplot...")
     if tasks_with_auc_data or tasks_with_f1_data:
          try:
              fig_box = plt.figure(figsize=(max(10, len(task_names) * 1.5), 7))
              auc_plot_data = [auc_distributions[name] for name in task_names if auc_distributions.get(name)]
              f1_plot_data = [f1_distributions[name] for name in task_names if f1_distributions.get(name)]
              # Ensure labels match the data being plotted
              plot_labels = [name for name in task_names if auc_distributions.get(name) or f1_distributions.get(name)]
              positions = np.arange(len(plot_labels)) * 2.0 # Positions for x-axis ticks

              if auc_plot_data:
                   bp_auc = plt.boxplot(auc_plot_data, positions=positions - 0.4, sym='', widths=0.6,
                                        patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
                                        medianprops=dict(color='darkblue'))

              if f1_plot_data:
                   bp_f1 = plt.boxplot(f1_plot_data, positions=positions + 0.4, sym='', widths=0.6,
                                       patch_artist=True, boxprops=dict(facecolor='lightcoral', color='red'),
                                       medianprops=dict(color='darkred'))

              plt.title("📦 Final Ensemble AUC (Blue) & F1 (Red) Spread per Task")
              plt.ylabel("Score")
              plt.xticks(positions, plot_labels, rotation=15, ha='right')
              plt.xlim(positions[0] - 1.5, positions[-1] + 1.5) # Adjust limits based on positions
              plt.grid(True, axis='y', linestyle='--')
              # Add legend manually
              plt.plot([], c='blue', label='AUC')
              plt.plot([], c='red', label='F1 Score')
              plt.legend()
              plt.tight_layout()

              if plots_dir:
                  os.makedirs(plots_dir, exist_ok=True)
                  filename = os.path.join(plots_dir, "dist_auc_f1_boxplot.png")
                  fig_box.savefig(filename, dpi=300, bbox_inches='tight')
                  print(f"  Saved combined boxplot to: {filename}")
                  plt.close(fig_box)
              else:
                  plt.pause(0.1)
          except Exception as e:
              print(f"  Error generating/saving combined boxplot: {e}")
              if 'fig_box' in locals() and fig_box is not None: plt.close(fig_box)
     else:
          print("No AUC or F1 data to plot for boxplot.")

     print("Post-training analysis complete.")