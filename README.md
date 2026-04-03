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

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/chaitanyasai11111/ISL_SoftComputingProject.git](https://github.com/chaitanyasai11111/ISL_SoftComputingProject.git)
   cd ISL_SoftComputingProject