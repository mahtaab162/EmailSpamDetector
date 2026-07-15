Overview

This project is a Machine Learning-based Email Spam Detection System that classifies emails as Spam or Not Spam (Ham). It uses Natural Language Processing (NLP) techniques to preprocess email text and a trained machine learning model to predict whether an email is legitimate or spam.
The application is built using Python, Scikit-learn, and Streamlit, providing an easy-to-use web interface for real-time predictions.

Features
Detects whether an email is Spam or Not Spam
User-friendly Streamlit web interface
Real-time email prediction
Text preprocessing using NLP techniques
TF-IDF Vectorization for feature extraction
Machine Learning classification model
Easy deployment on Streamlit Cloud

Technologies Used
Python
Scikit-learn
Pandas
NumPy
NLTK
Streamlit
Pickle

The project uses the SMS Spam Collection Dataset, containing labeled messages categorized as:

Spam
Ham (Not Spam)

Machine Learning Workflow
Load the dataset
Clean and preprocess the text
Tokenization
Remove punctuation
Remove stopwords
Apply stemming
Convert text into numerical features using TF-IDF Vectorizer
Train the Machine Learning model
Save the trained model and vectorizer using Pickle
Deploy using Streamlit

Author : Mahtaab Aalam
GitHub: [https://github.com/mahtaab162](http://localhost:8501)
