import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- PHASE 2: ARTIFICIAL NEURAL NETWORK TRAINING ---

def load_data():
    print("Loading extracted feature vectors and labels...")
    try:
        X_train = np.load('X_train_features.npy')
        X_val = np.load('X_val_features.npy')
        
        y_train = np.load('y_train.npy')
        y_val = np.load('y_val.npy')
        class_names = np.load('class_names.npy')
        
        return X_train, X_val, y_train, y_val, class_names
    except FileNotFoundError:
        print("Error: Could not find Phase 1 .npy files. Run phase1_prep.py first.")
        exit()

def build_mlp_model(input_dim, num_classes):
    print("Building the Multilayer Perceptron (MLP) Architecture...")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', name='Hidden_Layer_1'),
        Dense(64, activation='relu', name='Hidden_Layer_2'),
        Dense(32, activation='relu', name='Hidden_Layer_3'),
        Dense(num_classes, activation='softmax', name='Output_Layer')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.summary()
    return model

def plot_training_history(history):
    """Plots and saves the accuracy and loss curves for the project report."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='teal')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot Loss
    ax2.plot(history.history['loss'], label='Train Loss', color='teal')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Model Loss (Categorical Cross-Entropy)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('mlp_training_curves.png')
    print("\nSaved 'mlp_training_curves.png'. (Include this in your Chapter 5 Results!)")

if __name__ == "__main__":
    print("Initializing Phase 2: ANN Training...")
    
    # 1. Load Data (Notice we DO NOT load X_test here. It remains unseen!)
    X_train, X_val, y_train, y_val, class_names = load_data()
    num_classes = len(class_names)
    input_dim = X_train.shape[1] 
    
    # 2. Feature Scaling
    print("\nStandardizing feature scales...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler so we can use the exact same mathematical scaling in the final test
    joblib.dump(scaler, 'mlp_scaler.joblib')
    
    # 3. One-Hot Encoding Labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    
    # 4. Build Model
    model = build_mlp_model(input_dim, num_classes)
    
    # 5. Setup Callbacks
    checkpoint = ModelCheckpoint('best_mlp_model.weights.h5', 
                                 monitor='val_accuracy', 
                                 save_weights_only=True,
                                 save_best_only=True, 
                                 mode='max', 
                                 verbose=1)
    
    # 6. Train the Model
    print("\nStarting Training...")
    history = model.fit(X_train_scaled, y_train_cat,
                        validation_data=(X_val_scaled, y_val_cat),
                        epochs=50,
                        batch_size=32,
                        callbacks=[checkpoint])
    
    # 7. Generate Report Graphics
    plot_training_history(history)
    
    print("\nPhase 2 Complete! Weights saved as 'best_mlp_model.weights.h5' and Scaler saved as 'mlp_scaler.joblib'.")