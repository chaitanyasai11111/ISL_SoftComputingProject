import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score, confusion_matrix
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib
import os

print("Initializing Phase 4: Hybrid System Evaluation...")

# --- 1. LOAD DATA & ARTIFACTS ---
def load_test_environment():
    print("Loading Test Data and Saved Artifacts...")
    try:
        X_test = np.load('X_test_features.npy')
        y_test = np.load('y_test.npy')
        class_names = np.load('class_names.npy')
        scaler = joblib.load('mlp_scaler.joblib')
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure Phases 1 & 2 were run successfully.")
        exit()
        
    num_classes = len(class_names)
    
    # Rebuild MLP
    model = Sequential([
        Input(shape=(9,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.load_weights('best_mlp_model.weights.h5')
    
    return X_test, y_test, class_names, scaler, model

# --- 2. SETUP FUZZY SUPERVISOR ---
def build_fuzzy_supervisor():
    """Builds the rule blocks for ambiguity resolution."""
    # UV Block
    defect_depth = ctrl.Antecedent(np.arange(0, 1.05, 0.05), 'defect_depth')
    v_confidence = ctrl.Consequent(np.arange(0, 101, 1), 'v_confidence')
    
    defect_depth['shallow'] = fuzz.trimf(defect_depth.universe, [0, 0, 0.3])
    defect_depth['medium'] = fuzz.trimf(defect_depth.universe, [0.15, 0.4, 0.7])
    defect_depth['deep'] = fuzz.trimf(defect_depth.universe, [0.5, 1.0, 1.0])
    
    v_confidence['low'] = fuzz.trimf(v_confidence.universe, [0, 0, 40])
    v_confidence['medium'] = fuzz.trimf(v_confidence.universe, [20, 50, 80])
    v_confidence['high'] = fuzz.trimf(v_confidence.universe, [60, 100, 100])
    
    rule1 = ctrl.Rule(defect_depth['shallow'], v_confidence['low'])
    rule2 = ctrl.Rule(defect_depth['medium'], v_confidence['medium'])
    rule3 = ctrl.Rule(defect_depth['deep'], v_confidence['high'])
    
    uv_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3]))
    
    return uv_sim

# --- 3. EVALUATION LOOP ---
def run_hybrid_evaluation(X_test, y_test, class_names, scaler, model, uv_sim):
    print("\nStarting Hybrid Evaluation on Unseen Test Data...")
    
    X_test_scaled = scaler.transform(X_test)
    
    # Get raw MLP predictions all at once (much faster)
    print("Running Neural Network Inference...")
    mlp_probabilities = model.predict(X_test_scaled, verbose=0)
    
    mlp_predictions = []
    hybrid_predictions = []
    fuzzy_interventions = 0
    fuzzy_corrections = 0
    
    print("Applying Fuzzy Logic Supervisor to Ambiguous Cases...")
    for i in range(len(X_test)):
        # 1. Get MLP Top 2 predictions
        probs = mlp_probabilities[i]
        top_2_idx = np.argsort(probs)[-2:][::-1]
        top_2_classes = [class_names[idx] for idx in top_2_idx]
        
        mlp_pred_class = top_2_classes[0]
        actual_class = class_names[y_test[i]]
        
        mlp_predictions.append(mlp_pred_class)
        
        final_pred_class = mlp_pred_class
        
        # 2. Trigger Fuzzy Supervisor for U/V Cluster
        if 'U' in top_2_classes and 'V' in top_2_classes:
            fuzzy_interventions += 1
            
            # Extract depth feature (index 8)
            depth_val = X_test[i][8] 
            normalized_depth = min(depth_val / 30.0, 1.0) 
            
            uv_sim.input['defect_depth'] = normalized_depth
            uv_sim.compute()
            v_score = uv_sim.output['v_confidence']
            
            if v_score >= 60:
                final_pred_class = 'V'
            elif v_score <= 40:
                final_pred_class = 'U'
                
            # Track if Fuzzy actually fixed an MLP mistake
            if final_pred_class == actual_class and mlp_pred_class != actual_class:
                fuzzy_corrections += 1
                
        hybrid_predictions.append(final_pred_class)

    return mlp_predictions, hybrid_predictions, fuzzy_interventions, fuzzy_corrections

# --- 4. GENERATE REPORT GRAPHICS ---
def plot_final_confusion_matrix(y_true, y_pred, class_names):
    print("\nGenerating Final Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Hybrid System Confusion Matrix (MLP + Fuzzy Logic)', fontsize=16)
    plt.ylabel('Actual ISL Sign', fontsize=12)
    plt.xlabel('Predicted ISL Sign', fontsize=12)
    plt.tight_layout()
    
    plt.savefig('final_hybrid_confusion_matrix.png')
    print("Saved 'final_hybrid_confusion_matrix.png'.")

if __name__ == "__main__":
    X_test, y_test, class_names, scaler, model = load_test_environment()
    uv_sim = build_fuzzy_supervisor()
    
    mlp_preds, hybrid_preds, interventions, corrections = run_hybrid_evaluation(
        X_test, y_test, class_names, scaler, model, uv_sim
    )
    
    # Calculate Metrics
    y_true_classes = [class_names[idx] for idx in y_test]
    
    mlp_acc = accuracy_score(y_true_classes, mlp_preds) * 100
    hybrid_acc = accuracy_score(y_true_classes, hybrid_preds) * 100
    
    print("\n" + "="*50)
    print("FINAL ACADEMIC RESULTS FOR PROJECT REVIEW")
    print("="*50)
    print(f"Total Test Samples Evaluated : {len(y_test)}")
    print(f"Standalone MLP Accuracy      : {mlp_acc:.2f}%")
    print(f"Hybrid System Accuracy       : {hybrid_acc:.2f}%")
    print("-" * 50)
    print(f"Total Fuzzy Interventions    : {interventions}")
    print(f"Errors Corrected by Fuzzy    : {corrections}")
    print("="*50 + "\n")
    
    plot_final_confusion_matrix(y_true_classes, hybrid_preds, class_names)
    print("Phase 4 Complete. All artifacts are ready for your presentation.")