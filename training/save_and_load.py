import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict, Any # For type hinting
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

def save_results(filename: str,
                 vectorizer, # Or appropriate type hint
                 ensemble_models: Dict[str, List[Any]], # List contains fitted pipelines/estimators
                 task_names: List[str],
                 best_params: Dict[str, Optional[Dict]], # Params per task
                 # --- All Score Dictionaries ---
                 test_scores_auc: Dict[str, float],     # Mean test AUC per task
                 train_scores_auc: Dict[str, float],    # Mean train AUC per task
                 test_scores_f1: Dict[str, float],      # Mean test F1 per task
                 train_scores_f1: Dict[str, float],     # Mean train F1 per task
                 test_scores_bal_acc: Dict[str, float], # Mean test Bal Acc per task
                 train_scores_bal_acc: Dict[str, float],# Mean train Bal Acc per task
                 test_scores_mcc: Dict[str, float],     # Mean test MCC per task
                 train_scores_mcc: Dict[str, float],    # Mean train MCC per task
                 # --- Optimal Thresholds ---
                 optimal_thresholds: Dict[str, float]
                 ) -> None: # Function doesn't return anything explicitly

    print(f"\n--- Saving Results to {filename} ---")
    try:
        # Create the directory for the output file if it doesn't exist
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        elif output_dir:
             print(f"Output directory already exists: {output_dir}")
        else:
             print("Saving to current directory.")


        # Prepare the dictionary containing all data to be saved
        data_to_save = {
            # Core components
            "vectorizer": vectorizer,
            "ensemble_models": ensemble_models,
            "task_names": task_names,
            # Tuning results
            "best_params_per_task": best_params,
            # Evaluation metrics (Mean CV Scores)
            "mean_cv_test_scores_auc": test_scores_auc,
            "mean_cv_train_scores_auc": train_scores_auc,
            "mean_cv_test_scores_f1": test_scores_f1,
            "mean_cv_train_scores_f1": train_scores_f1,
            "mean_cv_test_scores_bal_acc": test_scores_bal_acc,
            "mean_cv_train_scores_bal_acc": train_scores_bal_acc,
            "mean_cv_test_scores_mcc": test_scores_mcc,
            "mean_cv_train_scores_mcc": train_scores_mcc,
            # Thresholds
            "optimal_thresholds": optimal_thresholds
        }

        # Save the dictionary using pickle
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f)

        print(f"✅ Save successful to '{filename}'.")

        # Report total models saved
        total_models = sum(len(models) for models in ensemble_models.values() if models is not None)
        print(f"🧠 Total trained models in saved ensemble structure: {total_models}")

    except PermissionError:
        print(f"⚠️ Error saving model file: Permission denied to write to '{filename}'. Check folder permissions.")
    except pickle.PicklingError as pe:
        print(f"⚠️ Error saving model file: Failed to pickle one of the objects. Error: {pe}")
        print("   Check if all objects (vectorizer, models, dictionaries) are pickleable.")
    except Exception as e:
        # Catch any other unexpected errors during saving
        import traceback
        print(f"⚠️ An unexpected error occurred during saving:")
        traceback.print_exc() # Print detailed traceback for unexpected errors

def load_and_prepare_data(data_path: str, task_names: List[str], 
                          random_state: int, EMBEDDING_MODEL: str
                          ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[SentenceTransformer]]:
    """Loads data, selects tasks, shuffles, and vectorizes text."""
    print(f"\n--- Loading and Preparing Data from {data_path} ---")
    try:
        df = pd.read_csv(data_path, index_col=0)
        df = df[task_names] # Select and order columns
        print(f"Loaded data shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Training data not found at {data_path}")
        return None, None, None
    except KeyError as e:
        print(f"Error: Column mismatch in CSV. Missing: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

    X_text = df.index.values
    Y = df.values

    # Shuffle data consistently
    print("Shuffling data...")
    np.random.seed(random_state)
    indices = np.random.permutation(len(X_text))
    X_text = X_text[indices]
    Y = Y[indices]

    print(f"Input text shape: {X_text.shape}")
    print(f"Target labels shape: {Y.shape}")
    if Y.shape[1] != len(task_names):
        print("Error: Mismatch between number of columns in Y and length of task_names.")
        return None, None, None

    # Vectorize input
    print(f"\nVectorizing input text using '{EMBEDDING_MODEL}'...")
    try:
        embedder_model = SentenceTransformer(EMBEDDING_MODEL)
        X_vec = embedder_model.encode(X_text, convert_to_tensor=False, show_progress_bar=True)
        print(f"Vectorized input shape: {X_vec.shape}")
        return X_vec, Y, embedder_model
    except Exception as e:
        print(f"Error during vectorization: {e}")
        return None, None, None