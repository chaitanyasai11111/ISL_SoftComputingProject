import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib
import random

print("Generating Annotated Result Images for Project Report...")

# Create results folder
if not os.path.exists('results'):
    os.makedirs('results')

# --- 1. LOAD MODELS & ARTIFACTS ---
class_names = np.load('class_names.npy')
num_classes = len(class_names)

model = Sequential([
    Input(shape=(9,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.load_weights('best_mlp_model.weights.h5')
scaler = joblib.load('mlp_scaler.joblib')

# --- 2. SETUP FUZZY SYSTEM ---
defect_depth = ctrl.Antecedent(np.arange(0, 1.05, 0.05), 'defect_depth')
v_confidence = ctrl.Consequent(np.arange(0, 101, 1), 'v_confidence')
defect_depth['shallow'] = fuzz.trimf(defect_depth.universe, [0, 0, 0.3])
defect_depth['medium'] = fuzz.trimf(defect_depth.universe, [0.15, 0.4, 0.7])
defect_depth['deep'] = fuzz.trimf(defect_depth.universe, [0.5, 1.0, 1.0])
v_confidence['low'] = fuzz.trimf(v_confidence.universe, [0, 0, 40])
v_confidence['medium'] = fuzz.trimf(v_confidence.universe, [20, 50, 80])
v_confidence['high'] = fuzz.trimf(v_confidence.universe, [60, 100, 100])
uv_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([
    ctrl.Rule(defect_depth['shallow'], v_confidence['low']),
    ctrl.Rule(defect_depth['medium'], v_confidence['medium']),
    ctrl.Rule(defect_depth['deep'], v_confidence['high'])
]))

# --- 3. FEATURE EXTRACTION ---
def extract_features(img_gray):
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary[0, 0] == 255:
        binary = cv2.bitwise_not(binary)
        
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0: return np.zeros(9)
        
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    hu = cv2.HuMoments(M).flatten()
    for i in range(7):
        if hu[i] != 0: hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i]))
            
    hull = cv2.convexHull(c, returnPoints=False)
    d_count, max_d = 0, 0.0
    if hull is not None and len(hull) > 3 and len(c) > 3:
        try:
            defects = cv2.convexityDefects(c, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    d = defects[i, 0][3] / 256.0
                    if d > 4.0: 
                        d_count += 1
                        max_d = max(max_d, d)
        except cv2.error: pass
            
    return np.nan_to_num(np.concatenate((hu, [d_count, max_d])))

# --- 4. PROCESS IMAGES ---
DATASET_PATH = 'Indian'
folders = sorted(os.listdir(DATASET_PATH))

for folder_name in folders:
    folder_path = os.path.join(DATASET_PATH, folder_name)
    if not os.path.isdir(folder_path): continue
        
    images = os.listdir(folder_path)
    # Pick 2 random images from each class folder
    selected_images = random.sample(images, 2)
    
    for img_name in selected_images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        # We need a bigger canvas to draw text cleanly, so we resize the output image
        display_img = cv2.resize(img, (256, 256))
        
        # Extract features from the standard 64x64 size
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (64, 64))
        
        features = extract_features(img_resized)
        if not np.any(features): continue
            
        features_scaled = scaler.transform([features])
        probs = model.predict(features_scaled, verbose=0)[0]
        
        top_2_idx = np.argsort(probs)[-2:][::-1]
        top_class = class_names[top_2_idx[0]]
        confidence = probs[top_2_idx[0]] * 100
        
        # Apply Fuzzy Logic
        override_text = ""
        top_2_names = [class_names[i] for i in top_2_idx]
        if 'U' in top_2_names and 'V' in top_2_names:
            uv_sim.input['defect_depth'] = min(features[8] / 30.0, 1.0)
            uv_sim.compute()
            v_score = uv_sim.output['v_confidence']
            if v_score >= 60:
                top_class, override_text = 'V', "(Fuzzy: V)"
            elif v_score <= 40:
                top_class, override_text = 'U', "(Fuzzy: U)"

        # Draw Text on the image
        color = (0, 255, 0) if top_class == folder_name else (0, 0, 255)
        cv2.putText(display_img, f"Pred: {top_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_img, f"True: {folder_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, f"{confidence:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if override_text:
            cv2.putText(display_img, override_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Save to results folder
        save_name = f"actual_{folder_name}_pred_{top_class}_{img_name}"
        cv2.imwrite(os.path.join('results', save_name), display_img)

print("Done! Check the 'results' folder for your generated images.")