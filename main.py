import pandas as pd
import numpy as np
import seaborn as sns
from preprocessing.Clean_Text import clean_text, tokenize_and_pad

# print(tf.__version__)
## Load the Dataset
df= pd.read_csv('data/News.csv')

# Display basic info
print("Dataset Columns:", df.columns)
print("Dataset Shape:", df.shape)
print(df.head())

# TO Ensure all text values are strings and replace NaN with empty strings
df['content'] = df['content'].astype(str).fillna('')

## APply the Text cleaning function
df['cleaned_content'] = df['content'].apply(clean_text)

# Tokenization and Padding
tokenizer, padded_sequences = tokenize_and_pad(df['cleaned_content'])

# Convert labels to numpy array
labels = np.array(df['label'])

# Print final processed shape
print("Tokenized and Padded Shape:", padded_sequences.shape)
print("Labels Shape:", labels.shape)
