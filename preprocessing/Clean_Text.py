import re
import nltk
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans input text by converting to lowercase, removing special characters, and stopwords."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def tokenize_and_pad(texts, max_length=200, vocab_size=5000):
    """Tokenizes and pads a list of texts to prepare them for model input."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")  # Keep top 5000 words
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return tokenizer, padded_sequences
