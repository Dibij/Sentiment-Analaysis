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

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# --------------------- Data Preprocessing --------------------- #
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stopwords_eng = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_eng]
    return " ".join(tokens)

# --------------------- Neural Network --------------------- #
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.layers = len(hidden_sizes) + 1  # Hidden layers + output layer
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Weight Initialization
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
        
    def activation(self, x, func='relu'):
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'tanh':
            return np.tanh(x)
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return x
    
    def activation_derivative(self, x, func='relu'):
        if func == 'relu':
            return (x > 0).astype(float)
        elif func == 'tanh':
            return 1 - np.tanh(x) ** 2
        return np.ones_like(x)
    
    def forward_propagation(self, X):
        activations = [X]
        pre_activations = []
        for i in range(self.layers):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            activation_func = 'softmax' if i == self.layers - 1 else 'relu'
            activations.append(self.activation(z, activation_func))
        return activations, pre_activations
    
    def backpropagation(self, activations, pre_activations, y_true):
        m = y_true.shape[0]
        gradients_w = [None] * self.layers
        gradients_b = [None] * self.layers
        
        # Output layer error
        dz = activations[-1] - y_true
        for i in reversed(range(self.layers)):
            gradients_w[i] = np.dot(activations[i].T, dz) / m
            gradients_b[i] = np.sum(dz, axis=0, keepdims=True) / m
            if i != 0:
                dz = np.dot(dz, self.weights[i].T) * self.activation_derivative(pre_activations[i-1])
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        for i in range(self.layers):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X, y, epochs=10, batch_size=32):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                activations, pre_activations = self.forward_propagation(X_batch)
                gradients_w, gradients_b = self.backpropagation(activations, pre_activations, y_batch)
                self.update_weights(gradients_w, gradients_b)
            
            # Compute accuracy
            y_pred = self.predict(X)
            acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.4f}")
    
    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]

# --------------------- Main Function --------------------- #
def main():
    # Load dataset
    data = pd.read_csv("Reviews.csv")
    data = data[['Text', 'Score']]
    data.dropna(inplace=True)
    
    # Preprocess text
    data['Text'] = data['Text'].apply(preprocess_text)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(data['Text']).toarray()
    y = to_categorical(data['Score'] - 1)  # Convert labels to one-hot
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train Model
    model = NeuralNetwork(input_size=5000, hidden_sizes=[256, 128, 64], output_size=5, learning_rate=0.001)
    model.train(X_train, y_train, epochs=10, batch_size=64)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
