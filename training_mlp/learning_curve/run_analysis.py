import numpy as np
from typing import List, Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
from sklearn.base import clone

from learning_curve.multilabel_learningcurve import manual_multilabel_learning_curve

# Assuming your learning curve module is accessible
try:
    from learning_curve.learn_curve_pipeline import plot_label_cooccurrence, plot_multilabel_learning_curve, learn_curve_per_label
    LEARNING_CURVE_MODULE_AVAILABLE = True
except ImportError:
    LEARNING_CURVE_MODULE_AVAILABLE = False
    print("Warning: learning_curve.learn_curve_pipeline module not found. Learning curve analysis will be skipped.")

def run_diagnostic_analysis(X_vec: np.ndarray, Y: np.ndarray, task_names: List[str],
                            base_pipeline: Pipeline, n_splits: int, random_state: int) -> Dict[str, Optional[float]]:
    """Runs co-occurrence and learning curve analysis (diagnostic only)."""
    print("\n--- Running Diagnostic Analysis ---")
    elbow_samples_lc_dict = {name: np.nan for name in task_names}

    if not LEARNING_CURVE_MODULE_AVAILABLE:
        print("Skipping diagnostic analysis as learning curve module is not available.")
        return elbow_samples_lc_dict

    # 1. Co-occurrence
    print("\nGenerating Label Co-occurrence Plot...")
    try:
        plot_label_cooccurrence(Y, task_names)
        plt.pause(0.1) # Allow plot to render without blocking
    except Exception as e:
        print(f"Could not generate co-occurrence plot: {e}")

    # 2. Per-Label Learning Curves
    print("\nGenerating Per-Label Learning Curves (ROC AUC)...")
    try:
        learning_curve_results = learn_curve_per_label(
            X_vec, Y, task_names,
            base_estimator_pipeline=base_pipeline,
            n_splits_lc=n_splits,
            random_state_lc=random_state
        )
        _, _, elbow_samples_lc_list = learning_curve_results
        elbow_samples_lc_dict = {name: val for name, val in zip(task_names, elbow_samples_lc_list)}
        print(f"Diagnostic Elbow points found (samples): {elbow_samples_lc_dict}")
    except Exception as e:
        print(f"Could not generate per-label learning curves: {e}")

    # 3. Multi-Label Learning Curves
    # 3. Multi-Label Learning Curves (Manual Approach)
    multi_output_pipeline_for_lc = MultiOutputClassifier(clone(base_pipeline), n_jobs=-1) # Can use n_jobs=-1 here

    # Call the manual function
    manual_multilabel_learning_curve(
        multi_output_pipeline_for_lc, X_vec, Y,
        cv_splits=n_splits, scoring_metric='jaccard_samples', random_state=random_state
    )
    manual_multilabel_learning_curve(
        multi_output_pipeline_for_lc, X_vec, Y,
        cv_splits=n_splits, scoring_metric='hamming_loss', random_state=random_state
    )
    # Add calls for other metrics if desired

    print("\nDiagnostic Analysis Complete.")
    return elbow_samples_lc_dict