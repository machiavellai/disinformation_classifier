import pandas as pd
import numpy as np
import seaborn as sns
from preprocessing.Clean_Text import clean_text, tokenize_and_pad
from preprocessing.Infer_labels import infer_labels
from Models.lstm_model import build_lstm_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample



# print(tf.__version__)
## Load the Dataset
df= pd.read_csv('data/News.csv')


# ✅ Apply Label Inference
df = infer_labels(df) 

df.to_csv("data/News_with_labels.csv", index=False)  # Save dataset with labels

# Display basic info
print("Dataset Columns:", df.columns)
print("Dataset Shape:", df.shape)
print(df.head())

# TO Ensure all text values are strings and replace NaN with empty strings
df['content'] = df['content'].astype(str).fillna('')

## APply the Text cleaning function
df['cleaned_content'] = df['content'].apply(clean_text)

# Balance the dataset
print("Label Distribution Before Balancing:", df['label'].value_counts())
df_fake = df[df['label'] == 1]  # Fake news
df_real = df[df['label'] == 0]  # Real news

# Upsample the minority class if needed
if len(df_real) > 0:
    df_real_upsampled = resample(df_real, 
                                 replace=True,    # Sample with replacement
                                 n_samples=len(df_fake),  # Match number of fake news
                                 random_state=42)  
    df_balanced = pd.concat([df_fake, df_real_upsampled])
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)  # Shuffle dataset
    df = df_balanced  # Use balanced dataset
    print("Balanced dataset saved. Label distribution:", df['label'].value_counts())
    df.to_csv("data/News_balanced.csv", index=False)
else:
    print("Warning: No real news (label 0) found in dataset. Consider adding more data.")


# Tokenization and Padding
tokenizer, padded_sequences = tokenize_and_pad(df['cleaned_content'])

# Convert labels to numpy array
labels = np.array(df['label'])

# Print final processed shape
print("Tokenized and Padded Shape:", padded_sequences.shape)
print("Labels Shape:", labels.shape)
print("Label Distribution Before Balancing:\n", df['label'].value_counts())


# Build the LSTM Model
vocab_size = 5000
embedding_dim = 128
max_length = 200
model = build_lstm_model(vocab_size, embedding_dim, max_length)

# Train the Model
model.fit(
    padded_sequences, labels,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)

# Save the model
model.save("models/lstm_disinfo_model.keras")
print("Model training completed and saved.")
