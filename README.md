# Product Category Classifier

This project automatically predicts product categories based on product titles using machine learning.

## Project Structure

- `data/products.csv` - dataset of product titles and categories
- `product_category_analysis.ipynb` - Jupyter notebook for data exploration and model evaluation
- `train_model.py` - script to train and save the final model
- `predict_category.py` - script to interactively predict category of a product
- `product_category_model.pkl` - saved trained model
- `vectorizer.pkl` - saved TF-IDF vectorizer

## How to Use

### Train the Model
```py
python train_model.py
```

### Run the model
```py
python predict_category.py
