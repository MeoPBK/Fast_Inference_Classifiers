import pickle
import numpy as np
import os
from typing import Dict, Tuple, Any

# === Configuration ===
# Use the filename saved by the MLP training script
MODEL_FILENAME = "./Classifier_Multilabel_Light/trained_models/model_multilabel_MLP_tuned_bootstrapped.pkl"

# === Load Model and Data ===
print(f"Loading model data from: {MODEL_FILENAME}")
if not os.path.exists(MODEL_FILENAME):
    print(f"Error: Model file not found at {MODEL_FILENAME}")
    exit()

try:
    with open(MODEL_FILENAME, "rb") as f:
        saved_data = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    # If you still get ModuleNotFound error here, the PKL file itself has the dependency.
    # You MUST regenerate the PKL file using the training script where all functions
    # are defined locally (not imported from custom modules like threshold_tuning.py).
    exit()

# --- Extract Components ---
print("\nExtracting components from saved file...")
try:
    # Assuming SentenceTransformer type hint is correct, adjust if needed
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = Any # Placeholder if not installed

    vectorizer: SentenceTransformer = saved_data["vectorizer"]
    ensemble_models: Dict[str, list] = saved_data["ensemble_models"] # Dict: {task_name: [model1, model2, ...]}
    task_names: list[str] = saved_data["task_names"]
    optimal_thresholds: Dict[str, float] = saved_data.get("optimal_thresholds", {}) # Load probability thresholds

    # Optionally load other metadata for display
    best_params = saved_data.get("best_params_per_task", {})
    test_scores_auc = saved_data.get("mean_cv_test_scores_auc", {})
    train_scores_auc = saved_data.get("mean_cv_train_scores_auc", {})
    test_scores_f1 = saved_data.get("mean_cv_test_scores_f1", {})
    train_scores_f1 = saved_data.get("mean_cv_train_scores_f1", {})
    # Add bal_acc and mcc if saved
    test_scores_bal_acc = saved_data.get("mean_cv_test_scores_bal_acc", {})
    train_scores_bal_acc = saved_data.get("mean_cv_train_scores_bal_acc", {})
    test_scores_mcc = saved_data.get("mean_cv_test_scores_mcc", {})
    train_scores_mcc = saved_data.get("mean_cv_train_scores_mcc", {})


    if not optimal_thresholds:
        print("Warning: Optimal thresholds not found in model file. Decisions will use default 0.5 threshold.")
    else:
        print("\nLoaded Optimal Probability Thresholds (based on maximizing F1 on OOF probabilities):")
        for task, thresh in optimal_thresholds.items():
            print(f"  - {task:<30}: {thresh:.4f}")

except KeyError as e:
    print(f"Error: Missing essential key in saved model file: {e}")
    print("Please ensure the model file was saved correctly by the updated MLP training script.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during component extraction: {e}")
    exit()


print("\nSuccessfully loaded:")
print(f"- Vectorizer type: {type(vectorizer)}")
print(f"- Task Names: {task_names}")
print(f"- Ensemble contains models for tasks: {list(ensemble_models.keys())}")

# --- Verify Task Name Consistency ---
if set(task_names) != set(ensemble_models.keys()):
    print("\nWarning: Task names loaded from 'task_names' key do not perfectly match the keys in 'ensemble_models' dictionary!")

# --- Display Loaded Metadata (Optional but helpful) ---
print("\n--- Model Metadata (Mean CV Scores) ---")
metadata_found = False
for task in task_names:
    print(f"\n  Task: {task}")
    # Display Test Scores
    score_test_auc = test_scores_auc.get(task, 'N/A')
    score_test_f1 = test_scores_f1.get(task, 'N/A')
    score_test_bal_acc = test_scores_bal_acc.get(task, 'N/A')
    score_test_mcc = test_scores_mcc.get(task, 'N/A')
    if any(s != 'N/A' for s in [score_test_auc, score_test_f1, score_test_bal_acc, score_test_mcc]):
        metadata_found = True
    print(f"    Test AUC    : {score_test_auc:.4f}" if isinstance(score_test_auc, float) else f"    Test AUC    : {score_test_auc}")
    print(f"    Test F1     : {score_test_f1:.4f}" if isinstance(score_test_f1, float) else f"    Test F1     : {score_test_f1}")
    print(f"    Test Bal Acc: {score_test_bal_acc:.4f}" if isinstance(score_test_bal_acc, float) else f"    Test Bal Acc: {score_test_bal_acc}")
    print(f"    Test MCC    : {score_test_mcc:.4f}" if isinstance(score_test_mcc, float) else f"    Test MCC    : {score_test_mcc}")

    # Display Train Scores (Optional)
    # ... (add similar printing for train scores if desired) ...

    # Display Best Params (Optional)
    params = best_params.get(task)
    if params:
        metadata_found = True
        print("    Best Params Found:")
        for p_name, p_val in params.items():
             print(f"      - {p_name}: {p_val}") # Print params directly
    elif task in best_params:
         print("    Best Params Found: None (Tuning may have been skipped)")

if not metadata_found:
     print("  (No additional metadata found in file or metadata keys missing)")
print("-" * 22)


# ----------------------------------------
# Decision-Maker Function (Updated for MLP)
# ----------------------------------------

def decision_maker(input_text: str) -> Tuple[Dict[str, float], Dict[str, bool]]:
    print(f"\n--- Making Prediction for: '{input_text}' ---")
    try:
        # Ensure input is a list for sentence-transformers encode
        x_vec = vectorizer.encode([input_text], convert_to_tensor=False)
        # x_vec should have shape (1, n_features)
    except Exception as e:
        print(f"Error during text vectorization: {e}")
        return {}, {} # Return empty dicts on error

    avg_probabilities: Dict[str, float] = {}
    decisions: Dict[str, bool] = {}

    for task_name in task_names:
        # Get the specific optimal probability threshold for this task
        # Default to 0.5 if threshold is missing for some reason
        task_threshold = optimal_thresholds.get(task_name, 0.5)
        task_models = ensemble_models.get(task_name, [])

        if not task_models:
            print(f"Warning: No models found for task '{task_name}'. Assigning default 0.5 probability and decision.")
            avg_probabilities[task_name] = 0.5 # Assign neutral probability
            decisions[task_name] = (0.5 > task_threshold) # Decision based on default prob vs threshold
            continue

        task_proba_scores = []
        for model_index, model in enumerate(task_models):
            try:
                # Use predict_proba for MLP pipelines
                # Get probability of the positive class (index 1)
                proba_score = model.predict_proba(x_vec)[0, 1]
                task_proba_scores.append(proba_score)
            except AttributeError:
                 print(f"Warning: Model {model_index} for task '{task_name}' lacks predict_proba. Skipping.")
            except Exception as e:
                print(f"Warning: Error predicting probability with model {model_index} for task '{task_name}': {e}. Skipping.")
                continue

        # Average the probabilities from all successful models
        if task_proba_scores:
            avg_probabilities[task_name] = np.mean(task_proba_scores)
        else:
            print(f"Warning: All models failed probability prediction for task '{task_name}'. Assigning neutral 0.5 probability.")
            avg_probabilities[task_name] = 0.5 # Assign neutral probability if all fail

        # --- Apply OPTIMAL probability threshold ---
        # Compare the average probability directly to the optimal threshold for this task
        decisions[task_name] = (avg_probabilities[task_name] > task_threshold)
        # --- END Apply ---

    return avg_probabilities, decisions

# ----------------------------------------
# Try It on a New Input
# ----------------------------------------

query = "where can I find a new parrot pet?"
# query = "what's the capital of France?"
# query = "summarize our previous conversation about the budget"

# Call decision_maker - it uses optimal_thresholds internally
avg_probabilities, decisions = decision_maker(query)

print("\n" + "="*40)
print("📝 Input Query:", query)
print("="*40)

print(f"\n📊 Average Predicted Probabilities & Decisions (Optimal Thresholds):")
if avg_probabilities:
    for task, prob in avg_probabilities.items():
        # Get the decision made using the optimal threshold
        decision_indicator = "-> TRUE" if decisions.get(task, False) else "-> FALSE"
        # Get the optimal threshold used for this task's decision (for display)
        optimal_thresh = optimal_thresholds.get(task, 0.5) # Use the same fallback
        # Print probability, decision indicator, and the threshold used for that decision
        print(f"  - {task:<30}: {prob:.4f} {decision_indicator} (Thresh={optimal_thresh:.4f})")
else:
    print("  No probabilities were calculated.")

print("\n✅ Final Decisions (using optimal thresholds):")
if decisions:
    for task, decision in decisions.items():
        print(f"  - {task:<30}: {decision}")
else:
    print("  No decisions were made.")
print("="*40)

# (Commented out FastAPI part remains the same)