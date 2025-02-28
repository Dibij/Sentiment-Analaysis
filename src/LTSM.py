import pandas as pd
import numpy as np
import re
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# Load dataset
data = pd.read_csv("Reviews.csv")
data.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName'], inplace=True)

# Preprocessing
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stopwords_eng = set(stopwords.words('english'))

for col in ['Summary', 'Text']:
    data[col] = data[col].astype(str).str.lower()
    data[col] = data[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    data[col] = data[col].apply(lambda x: word_tokenize(x))
    data[col] = data[col].apply(lambda x: [word for word in x if word not in stopwords_eng])
    data[col] = data[col].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    data[col] = data[col].apply(lambda x: " ".join(x))

# TF-IDF Transformation
X = data['Text']
y = data['Score']
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Resampling (Under + Over Sampling)
undersample = RandomUnderSampler(sampling_strategy={5: 110000}, random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)
smote = SMOTE(sampling_strategy={1: 75000, 2: 80000, 3: 65000, 4: 95000}, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_resampled, y_train_resampled)

y_train_balanced = to_categorical(y_train_balanced - 1)
y_test = to_categorical(y_test - 1)

# Convert Sparse Matrix to Dense
X_train_balanced_dense = X_train_balanced.toarray()
X_test_dense = X_test.toarray()

# Define Neural Network
def custom_nn(input_shape):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(5, activation='softmax')
    ])
    return model

# Initialize Model
nn_model = custom_nn(X_train_balanced_dense.shape[1])
optimizer = tf.keras.optimizers.Adam()

# Custom Training Loop (Explicit Backpropagation)
epochs = 5
batch_size = 16
best_loss = float('inf')

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    dataset = tf.data.Dataset.from_tensor_slices((X_train_balanced_dense, y_train_balanced))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    for step, (batch_x, batch_y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = nn_model(batch_x, training=True)
            loss = tf.keras.losses.categorical_crossentropy(batch_y, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, nn_model.trainable_variables))
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}")
    
    # Validation Loss
    val_predictions = nn_model(X_test_dense, training=False)
    val_loss = tf.keras.losses.categorical_crossentropy(y_test, val_predictions)
    val_loss = tf.reduce_mean(val_loss)
    print(f"Validation Loss: {val_loss.numpy():.4f}")
    
    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        nn_model.save("best_sentiment_nn_model.h5")
        print("Best model saved!")

# Final Model Save
nn_model.save("final_sentiment_nn_model.h5")
print("Final model saved!")
