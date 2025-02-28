import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using various metrics.

    Parameters:
    - model: Trained neural network model
    - X_test: Test feature set
    - y_test: True labels (one-hot encoded)

    Prints:
    - Accuracy, Precision, Recall, F1-score, and Confusion Matrix
    """

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    y_true_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels

    # Calculate metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

    # Print results
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true_labels, y_pred_labels))
    print("\nConfusion Matrix:\n", conf_matrix)

# --------------------- Data Preprocessing --------------------- #
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stopwords_eng = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_eng]
    return " ".join(tokens)


class ImprovedNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001, dropout_rate=0.3):
        self.layers = len(hidden_sizes) + 1
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        
        # Weight Initialization - He initialization for ReLU
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def dropout(self, X, training=True):
        if not training or self.dropout_rate == 0:
            return X
        mask = np.random.binomial(1, 1-self.dropout_rate, size=X.shape) / (1-self.dropout_rate)
        return X * mask
    
    def activation(self, x, func='relu'):
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'softmax':
            # Stable softmax implementation
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return x
    
    def activation_derivative(self, x, func='relu'):
        if func == 'relu':
            return (x > 0).astype(float)
        return np.ones_like(x)
    
    def forward_propagation(self, X, training=True):
        activations = [X]
        pre_activations = []
        
        # Hidden layers with ReLU and dropout
        for i in range(self.layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            a = self.activation(z, 'relu')
            if training:
                a = self.dropout(a, training)
            activations.append(a)
        
        # Output layer with softmax (no dropout)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z)
        activations.append(self.activation(z, 'softmax'))
        
        return activations, pre_activations
    
    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_probs = -np.log(np.maximum(y_pred, 1e-10)) * y_true
        loss = np.sum(log_probs) / m
        return loss
    
    def backpropagation(self, activations, pre_activations, y_true):
        m = y_true.shape[0]
        gradients_w = [None] * self.layers
        gradients_b = [None] * self.layers
        
        # Output layer error (derivative of cross-entropy with softmax)
        dz = activations[-1] - y_true
        
        # Backpropagate through the network
        for i in reversed(range(self.layers)):
            gradients_w[i] = np.dot(activations[i].T, dz) / m
            gradients_b[i] = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:  # No need to calculate dz for the input layer
                dz = np.dot(dz, self.weights[i].T) * self.activation_derivative(pre_activations[i-1], 'relu')
        
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        for i in range(self.layers):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=128, early_stopping=5):
        n_samples = X.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            # Mini-batch gradient descent
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass with dropout
                activations, pre_activations = self.forward_propagation(X_batch, training=True)
                
                # Calculate loss
                batch_loss = self.cross_entropy_loss(activations[-1], y_batch)
                total_loss += batch_loss * len(X_batch)
                
                # Backward pass and update weights
                gradients_w, gradients_b = self.backpropagation(activations, pre_activations, y_batch)
                self.update_weights(gradients_w, gradients_b)
            
            # Calculate average loss
            avg_loss = total_loss / n_samples
            
            # Evaluate on training set
            y_pred = self.predict(X)
            train_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
            
            # Evaluate on validation set if provided
            val_info = ""
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_acc = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
                val_loss = self.cross_entropy_loss(y_val_pred, y_val)
                val_info = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}{val_info}")
    
    def predict(self, X):
        # Forward pass without dropout for prediction
        activations, _ = self.forward_propagation(X, training=False)
        return activations[-1]

# Load dataset with stratified sampling
data = pd.read_csv("Reviews.csv", nrows=20000)
data = data[['Text', 'Score']]
data.dropna(inplace=True)
data['Text'] = data['Text'].apply(preprocess_text)

# Convert scores to 0-indexed classes (1-5 → 0-4)
data['Score_idx'] = data['Score'] - 1

# Stratified train-test split to maintain class distribution
X_train, X_test, y_train_idx, y_test_idx = train_test_split(
    data['Text'], data['Score_idx'], 
    test_size=0.2, random_state=42,
    stratify=data['Score_idx']  # This ensures balanced classes in test set
)

# TF-IDF Vectorization (after splitting to avoid data leakage)
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()

# One-hot encode target variables
y_train = to_categorical(y_train_idx, num_classes=5)
y_test = to_categorical(y_test_idx, num_classes=5)

# Fix class imbalance in training set
# First check class distribution
print("Original training set distribution:")
unique, counts = np.unique(y_train_idx, return_counts=True)
print(dict(zip(unique, counts)))

# Apply SMOTE for all classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_idx_balanced = smote.fit_resample(X_train, y_train_idx)

# Verify balanced distribution
print("Balanced training set distribution:")
unique, counts = np.unique(y_train_idx_balanced, return_counts=True)
print(dict(zip(unique, counts)))

# Convert back to one-hot encoding
y_train_balanced = to_categorical(y_train_idx_balanced, num_classes=5)

model = ImprovedNeuralNetwork(input_size=5000, hidden_sizes=[256, 128, 64], output_size=5, learning_rate=0.001)
model.train(X_train_balanced, y_train_balanced, epochs=10, batch_size=128)

evaluate_model(model, X_test, y_test)
