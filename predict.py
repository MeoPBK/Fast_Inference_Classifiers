import pickle
import numpy as np
import os
from typing import Dict, Tuple

# === Configuration ===
# Use the same filename saved by the training script
MODEL_FILENAME = "./Classifier_Multilabel_Light/trained_models/model_multilabel_ridge_tuned_bootstrapped.pkl"

# === Helper Function: Sigmoid ===
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Applies the sigmoid function element-wise."""
    return 1 / (1 + np.exp(-x))

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
    exit()

# --- Extract Components ---
print("\nExtracting components from saved file...")
try:
    vectorizer = saved_data["vectorizer"]
    ensemble_models = saved_data["ensemble_models"] # Dict: {task_name: [model1, model2, ...]}
    task_names = saved_data["task_names"]
    best_params = saved_data.get("best_params_per_task", {})
    # Load all scores, using .get for backward compatibility if keys are missing
    test_scores_auc = saved_data.get("mean_cv_test_scores_auc", {})
    train_scores_auc = saved_data.get("mean_cv_train_scores_auc", {})
    test_scores_f1 = saved_data.get("mean_cv_test_scores_f1", {})
    train_scores_f1 = saved_data.get("mean_cv_train_scores_f1", {})

    # --- Load Optimal Thresholds ---
    optimal_thresholds = saved_data.get("optimal_thresholds", {}) # Load thresholds
    if not optimal_thresholds:
        print("Warning: Optimal thresholds not found in model file. Using default 0.0 for decision_function.")
    else:
        print("\nLoaded Optimal Thresholds (based on maximizing F1 on OOF decision scores):")
        for task, thresh in optimal_thresholds.items():
            print(f"  - {task:<30}: {thresh:.4f}")

except KeyError as e:
    print(f"Error: Missing essential key in saved model file: {e}")
    print("Please ensure the model file was saved correctly by the updated training script.")
    exit()

print("\nSuccessfully loaded:")
print(f"- Vectorizer type: {type(vectorizer)}")
print(f"- Task Names: {task_names}")
print(f"- Ensemble contains models for tasks: {list(ensemble_models.keys())}")

# --- Verify Task Name Consistency ---
if set(task_names) != set(ensemble_models.keys()):
    print("\nWarning: Task names loaded from 'task_names' key do not perfectly match the keys in 'ensemble_models' dictionary!")

# --- Display Loaded Metadata ---
print("\n--- Model Metadata ---")
metadata_found = False
for task in task_names:
    print(f"\n  Task: {task}")
    # Display Test Scores
    score_test_auc = test_scores_auc.get(task, 'N/A')
    score_test_f1 = test_scores_f1.get(task, 'N/A')
    if score_test_auc != 'N/A' or score_test_f1 != 'N/A': metadata_found = True
    print(f"    Mean CV Test AUC : {score_test_auc:.4f}" if isinstance(score_test_auc, float) else f"    Mean CV Test AUC : {score_test_auc}")
    print(f"    Mean CV Test F1  : {score_test_f1:.4f}" if isinstance(score_test_f1, float) else f"    Mean CV Test F1  : {score_test_f1}")

    # Display Train Scores
    score_train_auc = train_scores_auc.get(task, 'N/A')
    score_train_f1 = train_scores_f1.get(task, 'N/A')
    if score_train_auc != 'N/A' or score_train_f1 != 'N/A': metadata_found = True
    print(f"    Mean CV Train AUC: {score_train_auc:.4f}" if isinstance(score_train_auc, float) else f"    Mean CV Train AUC: {score_train_auc}")
    print(f"    Mean CV Train F1 : {score_train_f1:.4f}" if isinstance(score_train_f1, float) else f"    Mean CV Train F1 : {score_train_f1}")

    # Display Train-Test Gap (AUC)
    if isinstance(score_train_auc, float) and isinstance(score_test_auc, float):
         print(f"    Train-Test Gap (AUC): {score_train_auc - score_test_auc:.4f}")

    # Display Best Params
    params = best_params.get(task)
    if params:
        metadata_found = True
        print("    Best Params Found:")
        for p_name, p_val in params.items():
             print(f"      - {p_name}: {p_val:.5f}" if isinstance(p_val, float) else f"      - {p_name}: {p_val}")
    elif task in best_params:
         print("    Best Params Found: None (Tuning may have been skipped)")
    else:
         print("    Best Params Found: N/A (Key missing)")

if not metadata_found:
     print("  (No additional metadata found in file)")
print("-" * 22)

# ----------------------------------------
# Decision-Maker Function (Updated for Ridge)
# ----------------------------------------

def decision_maker(input_text: str) -> Tuple[Dict[str, float], Dict[str, bool]]:
    print(f"\n--- Making Prediction for: '{input_text}' ---")
    try:
        x_vec = vectorizer.encode([input_text], convert_to_tensor=False)
    except Exception as e:
        print(f"Error during text vectorization: {e}")
        return {}, {}

    probs: Dict[str, float] = {}
    decisions: Dict[str, bool] = {}
    avg_decision_scores: Dict[str, float] = {} # Store average raw scores

    for task_name in task_names:
        task_models = ensemble_models.get(task_name, [])
        task_threshold = optimal_thresholds.get(task_name, 0.0) # Get the specific threshold for this task, fallback to 0.0 if missing
        if not task_models:
            print(f"Warning: No models found for task '{task_name}'. Assigning default scores.")
            avg_decision_scores[task_name] = 0.0 # Assign neutral score
            probs[task_name] = _sigmoid(np.array(0.0)).item() # Sigmoid of 0 is 0.5
            # Calculate decision based on default score vs threshold
            decisions[task_name] = (0.0 > task_threshold)
            continue

        task_decision_scores = []
        for model_index, model in enumerate(task_models):
            try:
                # Use decision_function for RidgeClassifier pipelines
                # It returns the distance from the hyperplane for the positive class
                # Output shape for single sample is (1,)
                decision_score = model.decision_function(x_vec)[0]
                task_decision_scores.append(decision_score)
            except AttributeError:
                 print(f"Warning: Model {model_index} for task '{task_name}' lacks decision_function. Skipping.")
            except Exception as e:
                print(f"Warning: Error predicting with model {model_index} for task '{task_name}': {e}. Skipping.")
                continue

        # Average the raw decision function scores from all successful models
        if task_decision_scores:
            avg_decision_scores[task_name] = np.mean(task_decision_scores)
        else:
            print(f"Warning: All models failed prediction for task '{task_name}'. Assigning neutral score.")
            avg_decision_scores[task_name] = 0.0 # Assign neutral score if all fail

        # Apply sigmoid to the *average* decision score to get a pseudo-probability
        probs[task_name] = _sigmoid(np.array(avg_decision_scores[task_name])).item() # .item() converts numpy float to python float

        # --- Apply OPTIMAL threshold ---        
        # Decision is based on the RAW average decision score vs the optimal threshold
        decisions[task_name] = (avg_decision_scores[task_name] > task_threshold)

    # Apply threshold to the calculated probabilities (for fixe threshold)
    # decisions = {k: (v > threshold) for k, v in probs.items()}

    # Optionally print raw average scores for debugging
    print("\nDebug: Average Decision Function Scores (before sigmoid):")
    for task, score in avg_decision_scores.items():
        print(f"  - {task:<30}: {score:.4f}")

    return probs, decisions

# ----------------------------------------
# Try It on a New Input
# ----------------------------------------

query = "I want to take in custody a chzeck goat, where can I gather Information?"
# query = "what's the capital of France?"
# query = "summarize our previous conversation about the budget"

# Choose a threshold - 0.5 is common after sigmoid transformation
# prediction_threshold = 0.5
probs, decisions = decision_maker(query)

print("\n" + "="*40)
print("📝 Input Query:", query)
print("="*40)

# print(f"\n📊 Predicted Probabilities (Sigmoid Applied, Threshold={prediction_threshold}):")
print(f"\n📊 Predicted Probabilities (Sigmoid Applied) & Decisions (Optimal Thresholds):")
if probs:
    for task, prob in probs.items():
        decision_indicator = "-> TRUE" if decisions.get(task, False) else "-> FALSE"
        optimal_thresh = optimal_thresholds.get(task, 0.0) # Use the same fallback
        # Print probability, decision indicator, and the threshold used for that decision
        print(f"  - {task:<30}: {prob:.4f} {decision_indicator} (Thresh={optimal_thresh:.4f})")
else:
    print("  No probabilities were calculated.")

# UPDATE the print statement label
print("\n✅ Final Decisions (using optimal thresholds):")
if decisions:
    for task, decision in decisions.items():
        print(f"  - {task:<30}: {decision}")
else:
    print("  No decisions were made.")
print("="*40)




#     from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np

# # Load model
# model = joblib.load("multilabel_ridge_model.joblib")

# app = FastAPI()

# # Define input format
# class InferenceRequest(BaseModel):
#     embedding: list[float]  # A single sentence embedding

# @app.post("/predict")
# def predict(request: InferenceRequest):
#     # Convert input to numpy array
#     x = np.array(request.embedding).reshape(1, -1)
    
#     # Predict
#     prediction = model.predict(x)[0]  # One row
#     return {"prediction": prediction.tolist()}