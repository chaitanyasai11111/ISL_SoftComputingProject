# Indian Sign Language (ISL) Classification using Soft Computing

This repository contains the code for an academic Soft Computing project aimed at classifying Indian Sign Language (ISL) alphabets and numbers (35 classes). 

To strictly adhere to the principles of Soft Computing and classical Machine Learning, this project actively avoids resource-intensive, "black-box" Deep Learning models (like Convolutional Neural Networks). Instead, it utilizes a hybrid approach combining **Geometric Feature Extraction**, an **Artificial Neural Network (MLP)**, and a **Fuzzy Inference System (FIS)**.

## 🧠 Core Architecture

The system pipeline consists of four main phases:

1. **Feature Extraction (Computer Vision):** Instead of processing raw pixels, we extract 9 highly specific mathematical features from the hand's silhouette:
   * **7 Hu Moments:** To describe the overall mass and geometry of the hand (translation, scale, and rotation invariant).
   * **2 Topological Metrics:** Convexity Defect count and Maximum Defect Depth to understand finger positioning.
2. **Artificial Neural Network (ANN):** A lightweight Multilayer Perceptron (MLP) trained on the 9 standardized features to output probability distributions across the 35 classes.
3. **Fuzzy Logic Supervisor:** A Mamdani Fuzzy Inference System containing expert, human-readable rules. It is triggered only when the ANN is confused by visually similar signs (e.g., distinguishing a 'U' from a 'V' based on the depth of the finger gap).
4. **Hybrid Evaluation:** A testing harness that runs unseen data through the ANN and applies the Fuzzy Supervisor to ambiguous clusters, proving that the hybrid system outperforms a standalone neural network.

### Note on "The Domain Gap"
Classical geometric features (like Hu Moments) evaluate the *entire* contour of an object. While they achieve extremely high accuracy on tightly cropped laboratory datasets, they are highly sensitive to real-world webcam environments (where wrists, forearms, and background shadows alter the mathematical geometry). This project successfully proves the efficacy of the proposed Soft Computing algorithms in a controlled environment, highlighting the exact mathematical reasons why the industry shifted to CNNs for real-time vision.

---
Install the required dependencies:

Bash
pip install opencv-python numpy scikit-learn tensorflow scikit-fuzzy matplotlib seaborn joblib
Add the Dataset:
Because of GitHub's file size limits, the raw image dataset is not included in this repository.

Create a folder named Indian in the root directory.

Place the ISL dataset images inside this folder (organized into subfolders A-Z and 1-9).

🚀 How to Run the Pipeline
Run the scripts in the following exact order to replicate the results:

Phase 1: Data Preparation & Feature Extraction
Reads the raw images, extracts the 9 mathematical features, splits the data (70% Train / 15% Val / 15% Test), and saves them as lightweight .npy arrays.

Bash
python phase1_prep.py
Phase 2: Neural Network Training
Standardizes the features and trains the 128-64-32 MLP on the training set, tuning it via the validation set. Saves the scaler and model weights.

Bash
python phase2_mlp_training.py
Phase 3: Fuzzy Supervisor Module
This script contains the FuzzySupervisor class (imported in Phase 4). You can run it standalone to generate the Membership Function graphs for your report.

Bash
python phase3_fuzzy_supervisor.py
Phase 4: Hybrid System Evaluation
Evaluates the unseen 15% Test Dataset through the entire hybrid pipeline (ANN + Fuzzy Logic). Outputs final hybrid accuracy and generates the overall Confusion Matrix.

Bash
python phase4_hybrid_evaluation.py
Optional: Generate Result Images
Picks random test images, classifies them through the hybrid pipeline, and overlays the true label, prediction, confidence, and Fuzzy logic overrides directly onto the image. Saved in the results/ folder.

Bash
python generate_result_images.py

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/chaitanyasai11111/ISL_SoftComputingProject.git](https://github.com/chaitanyasai11111/ISL_SoftComputingProject.git)
   cd ISL_SoftComputingProject
