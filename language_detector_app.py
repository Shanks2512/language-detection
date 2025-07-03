import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset (make sure 'language.csv' is in the same directory)
data = pd.read_csv(r"language.csv\language.csv")

# Check for nulls and data types (for development/debugging)
assert not data.isnull().sum().any(), "Dataset contains null values. Clean the dataset first."
assert 'Text' in data.columns and 'language' in data.columns, "Expected 'Text' and 'language' columns in dataset."

# Feature and target
x = np.array(data['Text'])
y = np.array(data['language'])

# Vectorize text data
cv = CountVectorizer()
x = cv.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(x_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'language_model.pkl')
joblib.dump(cv, 'vectorizer.pkl')

print(f"Model trained with accuracy: {model.score(x_test, y_test):.2f}")