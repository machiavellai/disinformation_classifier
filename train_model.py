import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('data.csv', encoding='ISO-8859-1')

# Debugging: Print initial dataset info
print("Initial dataset shape:", df.shape)
print(df.head())  # Check first few rows of the dataset

# Drop rows with missing values
print("Before dropna:", df.shape)
df = df.dropna()
print("After dropna:", df.shape)

# # Ensure no empty or invalid text entries
# df['Text'] = df['Text'].str.strip()  # Remove leading/trailing whitespace
# df = df[df['Text'] != '']  # Filter out empty strings

# Check class distribution
print("Class distribution:")
print(df['Label'].value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], df['Label'], test_size=0.2, random_state=42
)

# Debugging: Print sizes of train and test sets
print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Create pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Cross-validation to evaluate model performance
print("Performing cross-validation...")
cv_scores = cross_val_score(model, df['Text'], df['Label'], cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Evaluate model on test set
print("Evaluating model on test set...")
y_pred = model.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Debugging: Sample predictions
print("Sample predictions:")
for text, true_label, pred_label in zip(X_test[:5], y_test[:5], y_pred[:5]):
    print(f"Text: {text}\nTrue Label: {true_label}, Predicted Label: {pred_label}\n")

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report


# # Load dataset
# df = pd.read_csv('data.csv', encoding='ISO-8859-1')
# df['Text'] = df['Text'].str.strip()  # Remove leading/trailing whitespace
# df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex=True)  # Remove extra spaces

# print(df.head())
# print("Columns:", df.columns)

# print("Before dropna:", df.shape)
# # Drop rows with missing values
# df = df.dropna()
# print("After dropna:", df.shape)

# # Check the columns
# print(df['Label'].value_counts())

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.2, random_state=42)

# # Create pipeline
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# # Train model
# model.fit(X_train, y_train)


# # Evaluate model
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))