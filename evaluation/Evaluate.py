import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing.Clean_Text import tokenize_and_pad
from preprocessing.Infer_labels import infer_labels
import pandas as pd

# Add the root directory to sys.path


# Load the trained model
model = tf.keras.models.load_model("models/lstm_disinfo_model.keras")

# Load the test dataset
df = pd.read_csv("data/News_with_labels.csv")

# Ensure all text values are strings and replace NaN with empty strings
df['content'] = df['content'].astype(str).fillna('')

# Apply tokenization and padding
_, test_sequences = tokenize_and_pad(df['content'])

# Apply label inference if 'label' column is missing
if 'label' not in df.columns:
    print("Applying label inference...")
    df = infer_labels(df)

# Convert labels to numpy array
print("Dataset Columns:", df.columns)  # Debugging step
print("Dataset Shape:", df.shape)  # Debugging step
labels = np.array(df['label'])

# Make predictions
y_pred = model.predict(test_sequences)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class

# Evaluate the model
accuracy = accuracy_score(labels, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print("Label Distribution:\n", df['label'].value_counts())
print(classification_report(labels, y_pred))

# Confusion Matrix
cm = confusion_matrix(labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
