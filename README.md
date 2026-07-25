# 🧬 Protein Classification App

This project provides a complete machine learning pipeline and an interactive web application for classifying proteins into different categories (e.g., Structural, Receptor, Enzyme, Transport) based on their amino acid sequences and physicochemical properties.

## 📊 Dataset Reference

The data used to train and test the models in this project is the **Bioinformatics Protein Dataset (Simulated)**. 
You can find and download the original dataset on Kaggle here: 
👉 [Bioinformatics Protein Dataset (Simulated)](https://www.kaggle.com/datasets/gallo33henrique/bioinformatics-protein-dataset-simulated/data)

## ✨ Features

1. **Data Analysis & Modeling (`analysis_and_modeling.ipynb`)**
   - Exploratory Data Analysis (EDA) of the protein features.
   - Advanced feature engineering, including extracting 3-mers from protein sequences and applying TF-IDF vectorization.
   - Model training and hyperparameter tuning using `RandomizedSearchCV` for Logistic Regression, Random Forest, and XGBoost.
   - Saves the best-performing model and label encoder as `.pkl` files for deployment.

2. **Interactive Web Application (`app.py`)**
   - Built with **Streamlit** for a seamless user experience.
   - Allows users to upload new protein data in CSV format.
   - Automatically preprocesses the data and generates predictions using the trained model.
   - Displays interactive charts (Bar, Pie, and Scatter plots) using **Plotly** to visualize the distribution of predicted classes and feature relationships.
   - Provides an option to download the prediction results as a CSV file.

## 🚀 How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed. Then, install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Generate the Model
Before running the web app, you need to train the model and generate the `.pkl` files. 
Open the Jupyter Notebook and run all the cells:
```bash
jupyter notebook analysis_and_modeling.ipynb
```
*(This will create `best_protein_classifier.pkl` and `label_encoder.pkl` in your project directory).*

### 3. Run the Streamlit App
Once the model is saved, you can start the interactive web application by running:
```bash
streamlit run app.py
```
This will automatically open the app in your default web browser.

## 📁 Project Structure

- `data/` - Folder containing the training and testing CSV files.
- `analysis_and_modeling.ipynb` - Jupyter Notebook for EDA, preprocessing, and model training.
- `app.py` - The Streamlit web application.
- `requirements.txt` - List of Python dependencies required to run the project.
- `best_protein_classifier.pkl` - The trained machine learning pipeline (generated after running the notebook).
- `label_encoder.pkl` - The fitted label encoder for target classes (generated after running the notebook).
