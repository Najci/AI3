# predict_category.py

import joblib

# Load saved model and vectorizer
model = joblib.load('product_category_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

print("Product Category Predictor (type 'exit' to quit)")

while True:
    title = input("Enter product title: ")
    if title.lower() == 'exit':
        break
    X_vect = vectorizer.transform([title])
    prediction = model.predict(X_vect)
    print(f"Predicted Category: {prediction[0]}\n")
