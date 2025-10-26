# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load and clean data
df = pd.read_csv('data/products.csv')
df = df.dropna(subset=['Product Title', 'Category Label'])
df['Category Label'] = df['Category Label'].str.lower().str.strip()

# Split data
X = df['Product Title']
y = df['Category Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train final model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Save model and vectorizer
joblib.dump(model, 'product_category_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully!")
