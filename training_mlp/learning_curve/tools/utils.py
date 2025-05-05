import numpy as np
import warnings
from scipy.stats import norm
from scipy.signal import savgol_filter
from typing import Tuple, Optional, List

# --- Kneed Import ---
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
    print("Warning: 'kneed' library not found. Elbow detection will use max training size.")
    print("Install with: pip install kneed")

# ===========================================
# HELPER FUNCTIONS
# ===========================================

def confidence_interval_auc(scores: np.ndarray, confidence: float=0.95)->Tuple[float, float]:
    # Ensure scores is an array and handle potential NaNs from failed folds
    scores = np.array(scores)
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) < 2: # Need at least 2 points for std dev
        return np.nan, np.nan # Return NaNs if not enough data

    mean_auc = np.nanmean(scores) # Use nanmean
    std_err = np.nanstd(scores) / np.sqrt(len(valid_scores)) # Use nanstd and count of valid scores
    if std_err == 0: # Avoid error if all scores are identical
        return mean_auc, mean_auc
    h = std_err * norm.ppf(1 - (1 - confidence) / 2)
    return mean_auc - h, mean_auc + h



def detect_elbow_kneed(train_sizes_abs: np.ndarray, test_scores_mean: np.ndarray,
                        curve: str = 'concave', direction: str = 'increasing',
                        S: float = 1.0, min_points: int = 3,
                        window: int = 3, threshold: float = 0.0001,
                        use_kneed_fallback: bool = True, verbose: bool = True
                        ) -> int:
    """
    Detects the elbow using strict gradient thresholding.
    If no elbow is found, optionally falls back to Kneedle.
    """

    valid_indices = np.where(~np.isnan(test_scores_mean))[0]
    if len(valid_indices) == 0: # Handle case where all scores are NaN
         if verbose:
             print("     Elbow Detection: No valid points found.")
         # Return 0 or perhaps raise an error, returning max of empty array is problematic
         return int(train_sizes_abs[-1]) if len(train_sizes_abs) > 0 else 0

    x_points = train_sizes_abs[valid_indices]
    y_points = test_scores_mean[valid_indices]

    if len(x_points) < min_points:
        if verbose:
            print(f"     Elbow Detection: Not enough valid points ({len(x_points)} < {min_points}).")
        # Ensure x_points is not empty before accessing [-1]
        return int(x_points[-1]) if len(x_points) > 0 else (int(train_sizes_abs[-1]) if len(train_sizes_abs) > 0 else 0)

    # Optional smoothing
    y_smooth = y_points # Default to original points if smoothing fails or isn't needed
    if len(x_points) >= 5: # Savgol needs window_length < number of points
        try:
            # Ensure window length is odd and less than number of points
            smooth_window = min(5, len(y_points) // 2 * 2 + 1)
            if smooth_window >= 3: # Polyorder must be less than window length
                 y_smooth = savgol_filter(y_points, window_length=smooth_window, polyorder=2)
            else:
                 y_smooth = y_points # Not enough points for meaningful smoothing
        except Exception as e:
            if verbose:
                print(f"     Elbow Detection: Smoothing failed ({e}), using raw points.")
            y_smooth = y_points
    else:
        y_smooth = y_points # Not enough points for smoothing


    # Gradient-based flat spot detection
    try:
        # Use np.gradient on the potentially smoothed points
        gradient = np.gradient(y_smooth, x_points) # Use x_points for potentially uneven spacing
        below = gradient < threshold

        # Check for consecutive points below threshold
        for i in range(len(below) - window + 1):
            if np.all(below[i:i + window]):
                elbow_x = x_points[i] # Elbow is the *start* of the flat region
                if verbose:
                    print(f"     Elbow Detection (Gradient Method): Detected flat region starting at ~{int(elbow_x)} samples (gradient < {threshold} for {window} points).")
                return int(elbow_x)

    except Exception as e:
         if verbose:
             print(f"     Elbow Detection: Gradient calculation failed ({e}).")
         # Proceed to fallback if enabled


    # Fallback to Kneedle if no flat region found by gradient method
    if use_kneed_fallback and KNEED_AVAILABLE:
        try:
            # Use the original (non-smoothed) y_points for Kneedle as it has its own normalization/diff method
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning) # Kneedle can be verbose with warnings
                warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore potential runtime warnings from Kneedle internals
                kneedle = KneeLocator(
                    x=x_points,
                    y=y_points, # Use original y_points
                    curve=curve,
                    direction=direction,
                    S=S,
                    online=True # Often suitable for learning curves
                    )
            if kneedle.knee is not None:
                if verbose:
                    print(f"     Elbow Detection (Kneedle Fallback): Kneedle found elbow at ~{int(kneedle.knee)} samples.")
                return int(kneedle.knee)
            else:
                 if verbose:
                     print("     Elbow Detection (Kneedle Fallback): Kneedle did not find a knee point.")
        except Exception as e:
            if verbose:
                print(f"     Elbow Detection (Kneedle Fallback): Kneedle failed: {e}")

    # Final Fallback to max training size
    final_elbow = int(x_points[-1])
    if verbose:
        print(f"     Elbow Detection: No elbow found by configured methods, using max training samples: {final_elbow}")
    return final_elbow
