import numpy as np

def calculate_metrics(predictions, ground_truths):
    """
    predictions: (N, 3) array
    ground_truths: (N, 3) array
    """
    mse = np.mean((predictions - ground_truths)**2, axis=0) # [steer_mse, th_mse, br_mse]
    mae = np.mean(np.abs(predictions - ground_truths), axis=0)
    
    print(f"Metrics (Steering, Throttle, Brake):")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    
    return mse, mae

def evaluate_smoothness(steering_sequence):
    """
    Calculate smoothness of steering (jerk/derivative).
    Lower is smoother.
    """
    smoothness = np.mean(np.abs(diff))
    return smoothness

def discretize_signals(signals):
    """
    Helper to convert continuous controls to classes.
    signals: (N, 3) array [steer, throttle, brake]
    Returns: steer_cls, throttle_cls, brake_cls
    """
    steer = signals[:, 0]
    throttle = signals[:, 1]
    brake = signals[:, 2]
    
    # Steering: 0=Left, 1=Straight, 2=Right
    steer_cls = np.ones_like(steer, dtype=int)
    steer_cls[steer < -0.1] = 0
    steer_cls[steer > 0.1] = 2
    
    # Throttle: 0=Idle, 1=Accelerate
    throttle_cls = (throttle > 0.1).astype(int)
    
    # Brake: 0=Coast, 1=Brake
    brake_cls = (brake > 0.1).astype(int)
    
    return steer_cls, throttle_cls, brake_cls

def calculate_f1_score(predictions, ground_truths):
    """
    Calculate Macro F1 Score for driving controls.
    predictions: (N, 3) continuous
    ground_truths: (N, 3) continuous
    """
    from sklearn.metrics import f1_score
    
    pred_s, pred_t, pred_b = discretize_signals(predictions)
    gt_s, gt_t, gt_b = discretize_signals(ground_truths)
    
    # Use zero_division=0 to avoid warnings if some classes are missing in batch
    f1_s = f1_score(gt_s, pred_s, average='macro', zero_division=0)
    f1_t = f1_score(gt_t, pred_t, average='macro', zero_division=0)
    f1_b = f1_score(gt_b, pred_b, average='macro', zero_division=0)
    
    print(f"F1 Scores (Macro) - Steering: {f1_s:.4f} | Throttle: {f1_t:.4f} | Brake: {f1_b:.4f}")
    
    return f1_s, f1_t, f1_b
