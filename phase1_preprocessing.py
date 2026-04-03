import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATASET_PATH = 'Indian' 
IMAGE_SIZE = (64, 64)

def extract_features(img_gray):
    """
    Takes a grayscale image, extracts the hand contour, 
    and calculates Hu Moments and Convexity Defects.
    Returns a 1D numerical array (Feature Vector of 9 elements).
    """
    # 1. Segmentation (Thresholding)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # SAFETY CHECK: Ensure background is black and hand is white
    if binary[0, 0] == 255:
        binary = cv2.bitwise_not(binary)
        
    # 2. Morphological Operations
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Contour Extraction
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(9)
        
    c = max(contours, key=cv2.contourArea)
    
    # 4. Hu Moments Extraction
    M = cv2.moments(c)
    hu_moments = cv2.HuMoments(M).flatten()
    for i in range(0, 7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
            
    # 5. Convexity Defects
    hull = cv2.convexHull(c, returnPoints=False)
    defect_count = 0
    max_depth = 0.0
    
    if hull is not None and len(hull) > 3 and len(c) > 3:
        try:
            defects = cv2.convexityDefects(c, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0 
                    if depth > 4.0: 
                        defect_count += 1
                        if depth > max_depth:
                            max_depth = depth
        except cv2.error:
            pass 
            
    feature_vector = np.concatenate((hu_moments, [defect_count, max_depth]))
    return np.nan_to_num(feature_vector)

def load_and_extract_data():
    print("Initializing Data Acquisition and Feature Extraction...")
    
    features_list = []
    labels = []
    class_names = []
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find folder '{DATASET_PATH}'.")
        return None, None, None

    folders = sorted(os.listdir(DATASET_PATH))
    total_images_processed = 0
    
    for label_idx, folder_name in enumerate(folders):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if not os.path.isdir(folder_path): continue
            
        class_names.append(folder_name)
        image_files = os.listdir(folder_path)
        
        print(f"Processing Class '{folder_name}' ({len(image_files)} images)...")
        
        for image_name in image_files:
            img_path = os.path.join(folder_path, image_name)
            img = cv2.imread(img_path)
            if img is None: continue
                
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, IMAGE_SIZE)
            
            # Extract features immediately instead of storing raw images
            features = extract_features(img_resized)
            
            features_list.append(features)
            labels.append(label_idx)
            total_images_processed += 1

    print(f"\nTotal Images Processed and Extracted: {total_images_processed}")
    return np.array(features_list), np.array(labels), np.array(class_names)

def perform_data_splitting(X, y):
    """Splits data into 70% Train, 15% Val, 15% Test."""
    print("\nPerforming Train/Val/Test Split (70/15/15)...")
    
    # 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    # Split the Temp 50/50 into Validation and Testing
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    print(f"Training Set:   {len(X_train)} samples")
    print(f"Validation Set: {len(X_val)} samples")
    print(f"Testing Set:    {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    X, y, class_names = load_and_extract_data()
    
    if X is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = perform_data_splitting(X, y)
        
        print("\nSaving feature arrays to disk (.npy)...")
        np.save('X_train_features.npy', X_train)
        np.save('X_val_features.npy', X_val)
        np.save('X_test_features.npy', X_test)
        
        np.save('y_train.npy', y_train)
        np.save('y_val.npy', y_val)
        np.save('y_test.npy', y_test)
        
        np.save('class_names.npy', class_names)
        
        print("Phase 1 Complete! Data is cleaned, extracted, split, and ready for the Neural Network.")