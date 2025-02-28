import os
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data
def load_data():
    data = pd.read_csv("Reviews.csv", nrows=20000)[['Text', 'Score']].dropna()
    data['Text'] = data['Text'].apply(preprocess_text)
    data['Score_idx'] = data['Score'] - 1
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(data['Text']).toarray()
    y = data['Score_idx'].values
    return X, y

# Custom visualization callback
class TrainingVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, save_dir="plots/"):
        super().__init__()
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self.lrs = []
        self.save_dir = save_dir
        self.csv_file = os.path.join(save_dir, "training_log.csv")
        os.makedirs(save_dir, exist_ok=True)
        with open(self.csv_file, mode='w', newline='') as f:
            csv.writer(f).writerow(["epoch", "loss", "val_loss", "accuracy", "val_accuracy", "learning_rate"])
    
    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['accuracy'].append(logs.get('accuracy', 0))
        self.history['val_accuracy'].append(logs.get('val_accuracy', 0))
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(lr)
        with open(self.csv_file, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, logs.get('loss', 0), logs.get('val_loss', 0),
                                     logs.get('accuracy', 0), logs.get('val_accuracy', 0), lr])
        if (epoch + 1) % 5 == 0 or epoch == 0:
            self.save_plots(epoch)
    
    def save_plots(self, epoch_info):
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        ax[0].plot(self.history['accuracy'], label="Train Accuracy", marker='o')
        ax[0].plot(self.history['val_accuracy'], label="Validation Accuracy", marker='o')
        ax[0].set_title("Model Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        ax[1].plot(self.history['loss'], label="Train Loss", marker='o')
        ax[1].plot(self.history['val_loss'], label="Validation Loss", marker='o')
        ax[1].set_title("Model Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        if self.lrs:
            ax[2].plot(range(len(self.lrs)), self.lrs, marker='o', linestyle='--')
            ax[2].set_title("Learning Rate Schedule")
            ax[2].set_xlabel("Epoch")
            ax[2].set_ylabel("Learning Rate")

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_progress_epoch_{epoch_info}.png")
        plt.close()
# Build improved model
def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main execution
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    X_train_resampled, y_train_resampled = SMOTETomek(random_state=42).fit_resample(X_train_scaled, y_train)
    X_test_resampled, y_test_resampled = SMOTETomek(random_state=42).fit_resample(X_test_scaled, y_test)
    model = build_model(X_train_resampled.shape[1], len(np.unique(y)))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        TrainingVisualizer(save_dir="plots/")
    ]
    history = model.fit(X_train_resampled, y_train_resampled, epochs=30, batch_size=64, validation_split=0.2, callbacks=callbacks, verbose=1)
    y_pred_classes = np.argmax(model.predict(X_test_resampled), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test_resampled, y_pred_classes))
    cm = confusion_matrix(y_test_resampled, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    print("\nTraining complete. Results saved.")

if __name__ == "__main__":
    main()
