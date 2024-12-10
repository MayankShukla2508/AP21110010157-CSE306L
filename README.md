# Intrusion Detection System Using Machine Learning

This project implements an Intrusion Detection System (IDS) using machine learning and deep learning techniques. The goal is to detect malicious activity in a network by leveraging various algorithms and evaluating their performance.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
Intrusion Detection Systems are critical for identifying unauthorized or malicious activities in networks. This project explores various machine learning and deep learning methods to classify network traffic as either normal or malicious. The UNSW-NB15 dataset, a well-known dataset in network security, is used for evaluation.

---

## Features
- Data preprocessing with PCA for dimensionality reduction.
- Implementation of machine learning algorithms:
  - Gradient Boosted Trees (GBT)
  - Expectation Maximization (EM)
  - Naive Bayes
  - Logistic Regression
  - k-Nearest Neighbors (KNN)
  - Random Forest
  - Support Vector Machines (SVM)
- Deep learning models:
  - Convolutional Neural Networks (CNN)
  - Artificial Neural Networks (ANN)
- Comparative analysis of algorithms based on accuracy and performance metrics.

---

## Dataset
The project uses the **UNSW-NB15 dataset**, which contains normal and malicious network activity.  
- **Training Dataset:** `UNSW_NB15_training-set.csv`  
- **Testing Dataset:** `UNSW_NB15_testing-set.csv`  

### Features of the Dataset:
- **Number of Records:** 2,540,044  
- **Target Variable:** `label` (0 for normal, 1 for malicious)  

---

## Algorithms Implemented
1. **Gradient Boosted Trees (GBT):** A powerful ensemble method.
2. **Expectation Maximization (EM):** A probabilistic clustering algorithm.
3. **Naive Bayes:** A simple probabilistic classifier.
4. **Logistic Regression:** A linear model for binary classification.
5. **k-Nearest Neighbors (KNN):** A distance-based classification technique.
6. **Random Forest:** An ensemble of decision trees.
7. **Support Vector Machines (SVM):** A robust classifier for linear and non-linear data.
8. **Convolutional Neural Networks (CNN):** A deep learning model for structured data.
9. **Artificial Neural Networks (ANN):** A multi-layer perceptron-based model.

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/IDS-MachineLearning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd IDS-MachineLearning
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Upload the dataset files (`UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`) to your working directory.  
2. Run the preprocessing script to prepare the dataset:
   ```bash
   python preprocess.py
   ```
3. Run individual algorithm files to train and test models:
   ```bash
   python gradient_boosted_trees.py
   python cnn_model.py
   python svm_model.py
   # Add others as needed
   ```

---

## Results
The performance of the models was evaluated using accuracy. Below is a summary:  

| Algorithm          | Accuracy (%) |
|--------------------|--------------|
| Gradient Boosted Trees (GBT) | **94.8**   |
| Expectation Maximization (EM) | 55.2       |
| Naive Bayes        | 62.3         |
| Logistic Regression| 85.4         |
| k-Nearest Neighbors (KNN) | 88.7         |
| Random Forest      | **93.5**     |
| Support Vector Machines (SVM) | **91.2** |
| Convolutional Neural Networks (CNN) | **95.4** |
| Artificial Neural Networks (ANN) | **93.9** |

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For significant changes, please open an issue first to discuss your ideas.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
Special thanks to the creators of the UNSW-NB15 dataset and the open-source community for their invaluable tools and resources.
