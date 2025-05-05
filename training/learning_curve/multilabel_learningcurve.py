# Example Sketch (add this function to fA3 or main script)
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score, hamming_loss # Import metrics
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt

def manual_multilabel_learning_curve(estimator, X, Y, cv_splits=5,
                                     train_sizes=np.linspace(0.1, 1.0, 10),
                                     scoring_metric='jaccard_samples', random_state=42):
    """Manually calculates learning curve scores for multi-label tasks."""
    print(f"\n📊 Manually Generating Multi-Label Learning Curve ({scoring_metric})...")

    # --- Define Scorer ---
    scorer_func = None
    greater_is_better = True
    if scoring_metric == 'hamming_loss':
        scorer_func = hamming_loss
        greater_is_better = False # Lower is better
    elif scoring_metric == 'jaccard_samples':
        # Need average='samples' for jaccard_score in multi-label sample-wise
        scorer_func = lambda y_true, y_pred: jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    # Add other metrics like f1_score variants if needed
    # elif scoring_metric == 'f1_samples':
    #     scorer_func = lambda y_true, y_pred: f1_score(y_true, y_pred, average='samples', zero_division=0)
    else:
        print(f"Error: Manual implementation for scoring '{scoring_metric}' not added.")
        return

    # --- Prepare CV and Data Sizes ---
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    n_samples = X.shape[0]
    absolute_train_sizes = (train_sizes * n_samples).astype(int)
    # Ensure minimum size is reasonable (e.g., at least n_splits)
    absolute_train_sizes = np.maximum(absolute_train_sizes, cv_splits)
    absolute_train_sizes = np.unique(absolute_train_sizes) # Remove duplicates

    print(f"   Using train sizes: {absolute_train_sizes}")
    all_train_scores = []
    all_test_scores = []

    # --- Loop through Training Sizes ---
    for n_train_samples in absolute_train_sizes:
        print(f"    Processing size: {n_train_samples}")
        current_size_train_scores = []
        current_size_test_scores = []

        # --- Loop through CV Folds ---
        for fold, (train_idx_full, test_idx) in enumerate(cv.split(X, Y)):
            # Select the subset for this training size *from the current fold's train indices*
            # Ensure random subsetting if n_train_samples < len(train_idx_full)
            if n_train_samples >= len(train_idx_full):
                 train_idx_subset = train_idx_full
            else:
                 np.random.seed(random_state + fold + n_train_samples) # Seed for reproducibility
                 train_idx_subset = np.random.choice(train_idx_full, n_train_samples, replace=False)

            X_train, Y_train = X[train_idx_subset], Y[train_idx_subset]
            X_test, Y_test = X[test_idx], Y[test_idx]

            # Clone and fit the estimator
            estimator_clone = clone(estimator)
            try:
                estimator_clone.fit(X_train, Y_train)

                # Predict and Score on Test Set
                Y_pred_test = estimator_clone.predict(X_test)
                test_score = scorer_func(Y_test, Y_pred_test)
                current_size_test_scores.append(test_score)

                # Predict and Score on Train Set (Optional)
                Y_pred_train = estimator_clone.predict(X_train)
                train_score = scorer_func(Y_train, Y_pred_train)
                current_size_train_scores.append(train_score)

            except Exception as e:
                print(f"      Error in fold {fold+1} for size {n_train_samples}: {e}")
                current_size_test_scores.append(np.nan)
                current_size_train_scores.append(np.nan)
                continue # Skip to next fold

        # Average scores for this training size
        all_test_scores.append(np.nanmean(current_size_test_scores))
        all_train_scores.append(np.nanmean(current_size_train_scores))

    # --- Convert to Arrays and Plot ---
    all_train_scores = np.array(all_train_scores)
    all_test_scores = np.array(all_test_scores)

    plt.figure(figsize=(8, 5))
    plt.title(f"Manual Multi-Label Learning Curve ({scoring_metric})")
    plt.xlabel("Training examples")
    plt.ylabel(f"Score ({scoring_metric}) {' (Higher is Better)' if greater_is_better else ' (Lower is Better)'}")
    plt.grid()

    # Basic plotting (add std dev bands if needed by collecting all fold scores)
    plt.plot(absolute_train_sizes, all_train_scores, 'o-', color="r", label="Training score")
    plt.plot(absolute_train_sizes, all_test_scores, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.pause(0.1) # Non-blocking