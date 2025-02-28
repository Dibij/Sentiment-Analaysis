# **Sentiment Analysis Model**  
*This project is a work in progress. Accuracy metrics aren’t where I'd like them to be, but it's functional for now. I severely underestimated NLP, so I’ll revisit it in the future.*  

---

## **Table of Contents**  
1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Model Architecture](#model-architecture)  
6. [Dataset](#dataset)  
7. [Training Details](#training-details)  
8. [Results](#results)  
9. [Future Improvements](#future-improvements)  
10. [Contributing](#contributing)  

---

## **Overview**  
This project implements sentiment analysis using various techniques, including:  
- **LSTMs (initially)**  
- **Custom NumPy-based neural networks** with manual backpropagation  
- **Different data preprocessing and balancing strategies**  

It classifies sentiment into **positive, negative, or neutral** categories. Initially, the model was built with TensorFlow LSTMs but later evolved into a fully custom neural network for better control over training dynamics.  

---

## **Features**  

### **1. Data Handling & Preprocessing**  
- Loads a dataset containing text reviews and ratings.  
- Drops unnecessary columns (`Id`, `ProductId`, `UserId`, `ProfileName`).  
- Converts text to lowercase for uniformity.  
- Removes punctuation and special characters using regex.  
- Tokenizes text into words and removes English stopwords.  
- Applies **lemmatization** (e.g., "running" → "run").  

### **2. TF-IDF Transformation**  
- Converts text into numerical format using **TF-IDF (Term Frequency-Inverse Document Frequency)**.  
- Limits the vocabulary size to **5,000 most important words** for efficiency.  

### **3. Data Splitting & Balancing**  
- Splits data into training and testing sets with **stratification** (ensuring class balance).  
- Uses **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for underrepresented classes.  

### **4. Neural Network Architectures**  
#### **LSTM Model (Initially Implemented)**  
- **Embedding layer** to convert words into dense numerical vectors.  
- **Bidirectional LSTM** for better contextual understanding.  
- **Dropout layers** to prevent overfitting.  
- **Dense layers** with **softmax activation** for classification.  
- **Custom training loop** using TensorFlow’s `GradientTape` instead of `.fit()`.  

#### **Custom NumPy-Based Neural Network (Final Model)**  
- Built **from scratch** with:  
  - **Manual forward propagation** (fully connected layers).  
  - **Activation functions**: ReLU, Tanh, Softmax.  
  - **Backpropagation implemented manually**.  
  - **He-initialization** for better convergence.  
  - **Dropout regularization** to reduce overfitting.  
  - **Mini-batch training** for improved performance.  
- Fully removes TensorFlow dependencies.  

### **5. Advanced Modifications (V2 & V3)**  
- **Stable softmax** to prevent numerical instability.  
- **Early stopping** to prevent unnecessary training cycles.  
- **Stratified train-test splits** to maintain class balance.  
- **Batch normalization** for better weight distribution.  
- **Learning rate scheduling** to improve training stability.  
- **Training visualizations** with auto-logging features.  

---

## **Installation**  
### **Prerequisites:**  
- Python 3.x  
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `nltk`, `tensorflow` (for earlier versions)  

Install dependencies:  
```bash
pip install -r requirements.txt
```  

---

## **Usage**  
Run the script with:  
```bash
python main.py --input "Your text here"
```  

---

## **Model Architecture**  

### **1. LSTM (Long Short-Term Memory)**
- Designed to handle sequential data and capture long-range dependencies.  
- Uses a **gating mechanism** to control information flow.  
- Initially implemented but later **replaced for efficiency reasons**.  

### **2. Custom NumPy Neural Network**  
- Implemented using NumPy with:  
  - **He initialization** for better weight scaling.  
  - **ReLU activation** in hidden layers.  
  - **Softmax activation** in the output layer for classification.  
  - **Dropout Regularization** to prevent overfitting.  
  - **Mini-batch Gradient Descent** for efficiency.  
  - **Early Stopping** to avoid unnecessary training cycles.  

---

## **Dataset**  
- **Name:** Amazon Reviews  
- **Size:** ~500,000 samples (20,000 used for faster training).  
- **Sentiment Classes:**  
  - **Positive (Ratings: 4,5)**  
  - **Negative (Ratings: 1,2)**  
  - **Neutral (Rating: 3)**  
- **Preprocessing:** Tokenization, stopword removal, TF-IDF conversion.  

---

## **Training Details**  
- **Optimizer:** (SGD, Adam, manual backpropagation)  
- **Loss Function:** (Sparse Categorical Crossentropy, MSE)  
- **Training Time:** ~6 Minutes  
- **Evaluation Metrics:**  
  - Precision: **0.5067**  
  - Recall: **0.4891**  
  - F1 Score: **0.4768**  
  - Accuracy: **0.49**  

---

## **Results**  
### **Classification Report:**  
```
              precision    recall  f1-score   support

           0       0.62      0.50      0.55      3143
           1       0.49      0.34      0.40      3143
           2       0.62      0.49      0.54      3143
           3       0.39      0.27      0.32      3141
           4       0.42      0.84      0.56      3141

    accuracy                           0.49     15711
   macro avg       0.51      0.49      0.48     15711
weighted avg       0.51      0.49      0.48     15711
```  

### **Confusion Matrix:**  
```
[[1571  696  200  102  574]
 [ 482 1083  567  601  410]
 [ 343  321 1526  371  582]
 [  56   72  109  859 2045]
 [  89   34   75  297 2646]]
```  

---

## **Future Improvements**  
- **Expand dataset** with more variations.  
- Experiment with **transformer-based models** (e.g., BERT).  
- Optimize **manual backpropagation** further.  
- Convert the model into an **API for real-time sentiment analysis**.  

---

## **Contributing**  
Feel free to contribute by raising issues or submitting pull requests!  

