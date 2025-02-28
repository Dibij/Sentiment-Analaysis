import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to load your data
def load_data():
    # Load data from CSV
    data = pd.read_csv("Reviews.csv", nrows=20000)
    data = data[['Text', 'Score']]
    data.dropna(inplace=True)
    
    # Preprocess text
    data['Text'] = data['Text'].apply(preprocess_text)
    
    # Convert scores to zero-based indices (1-5 -> 0-4)
    data['Score_idx'] = data['Score'] - 1
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(data['Text']).toarray()
    
    # Get labels
    y = data['Score_idx'].values
    
    return X, y

# Load dataset
X, y = load_data()
print("Original dataset shape:", X.shape, y.shape)
print("Original class distribution:")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# Train-test split (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\nTraining set class distribution before SMOTE:")
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

print("\nTest set class distribution before SMOTE:")
unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to both training and testing data
# For training data
smote_train = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_train.fit_resample(X_train_scaled, y_train)

# For testing data (this is not standard practice but requested)
smote_test = SMOTETomek(random_state=42)
X_test_resampled, y_test_resampled = smote_test.fit_resample(X_test_scaled, y_test)

print("\nTraining set class distribution after SMOTE:")
unique, counts = np.unique(y_train_resampled, return_counts=True)
print(dict(zip(unique, counts)))

print("\nTest set class distribution after SMOTE:")
unique, counts = np.unique(y_test_resampled, return_counts=True)
print(dict(zip(unique, counts)))

# Build an improved model architecture
def build_improved_model(input_shape, num_classes):
    model = Sequential([
        # First hidden layer
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second hidden layer
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third hidden layer
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Use a better optimizer with learning rate scheduling
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Using sparse since labels are not one-hot encoded
        metrics=['accuracy']
    )
    
    return model

# Initialize and train the improved model
num_classes = len(np.unique(y))
model = build_improved_model(X_train_resampled.shape[1], num_classes)

# Set up callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Train the model with the balanced dataset
print("\nTraining the improved model:")
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model on the balanced test set
print("\nEvaluating on balanced test set:")
test_loss, test_acc = model.evaluate(X_test_resampled, y_test_resampled)
print(f"Test Accuracy: {test_acc:.4f}")

# Predict on the balanced test set
y_pred = model.predict(X_test_resampled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
precision = precision_score(y_test_resampled, y_pred_classes, average='weighted')
recall = recall_score(y_test_resampled, y_pred_classes, average='weighted')
f1 = f1_score(y_test_resampled, y_pred_classes, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_resampled, y_pred_classes))

# Print confusion matrix
cm = confusion_matrix(y_test_resampled, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Visualize training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(num_classes),
            yticklabels=range(num_classes))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Try with original test data for a fair evaluation
print("\nEvaluating on original imbalanced test data (more realistic scenario):")
y_pred_orig = model.predict(X_test_scaled)
y_pred_orig_classes = np.argmax(y_pred_orig, axis=1)

# Calculate metrics on original test data
precision_orig = precision_score(y_test, y_pred_orig_classes, average='weighted')
recall_orig = recall_score(y_test, y_pred_orig_classes, average='weighted')
f1_orig = f1_score(y_test, y_pred_orig_classes, average='weighted')

print(f"Precision (original test): {precision_orig:.4f}")
print(f"Recall (original test): {recall_orig:.4f}")
print(f"F1 Score (original test): {f1_orig:.4f}")

print("\nClassification Report (original test):")
print(classification_report(y_test, y_pred_orig_classes))

print("\nCompleted all evaluations and saved visualizations.")
