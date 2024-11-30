# Spam Detection using Multinomial Naive Bayes
This project implements a spam detection model using machine learning techniques, specifically the Multinomial Naive Bayes classifier with TF-IDF Vectorization. The model classifies messages into two categories: Spam and Not Spam (Ham).

The dataset used in this project consists of labeled SMS messages, which are preprocessed and then used to train and evaluate the model.

## Overview
This project demonstrates how to use machine learning for text classification. The key steps are:

Data Preprocessing: The raw SMS data is cleaned and tokenized.
Modeling: The Multinomial Naive Bayes algorithm is trained on the preprocessed data.
Evaluation: The performance of the model is evaluated using accuracy and precision scores.
Model Deployment: The trained model and TF-IDF vectorizer are saved for future use.
The model achieves a high precision score of 0.99346 for spam classification, indicating its effectiveness in identifying spam messages.

## Dataset
The dataset used for training the model is a collection of labeled SMS messages. The spam.csv file is loaded from a URL and contains two columns:

v1: The label (spam/ham)
v2: The message content

##
Certainly! Below is a more detailed and tailored README for your GitHub repository:

Spam Detection using Multinomial Naive Bayes
This project implements a spam detection model using machine learning techniques, specifically the Multinomial Naive Bayes classifier with TF-IDF Vectorization. The model classifies messages into two categories: Spam and Not Spam (Ham).

The dataset used in this project consists of labeled SMS messages, which are preprocessed and then used to train and evaluate the model.

Table of Contents
Overview
Getting Started
Project Structure
Model Details
Evaluation
Requirements
Usage
License
Acknowledgements
Overview
This project demonstrates how to use machine learning for text classification. The key steps are:

Data Preprocessing: The raw SMS data is cleaned and tokenized.
Modeling: The Multinomial Naive Bayes algorithm is trained on the preprocessed data.
Evaluation: The performance of the model is evaluated using accuracy and precision scores.
Model Deployment: The trained model and TF-IDF vectorizer are saved for future use.
The model achieves a high precision score for spam classification, indicating its effectiveness in identifying spam messages.

Getting Started
To get started with the project, follow these steps:

Prerequisites
You need Python and the following libraries installed:

pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
If you don't have these libraries, you can install them by running the following:

bash
Copy code
pip install -r requirements.txt
Alternatively, you can install each package individually with pip:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn nltk
Dataset
The dataset used for training the model is a collection of labeled SMS messages. The spam.csv file is loaded from a URL and contains two columns:

v1: The label (spam/ham)
v2: The message content
Project Structure
bash
Copy code
spam-detection/
├── README.md                     # Project documentation (this file)
├── spam-detection-model.ipynb     # Jupyter notebook with the model code
├── requirements.txt               # Required Python libraries
├── model.pkl                      # Pickled trained model (Multinomial Naive Bayes)
├── tfidf-vectorizer.pkl           # Pickled TF-IDF vectorizer
└── spam.csv                       # Dataset file (raw data)
Model Details
The model is built using the following key steps:

Data Preprocessing:

The text is converted to lowercase.
Non-alphabetic characters are removed.
Words are tokenized and stemmed.
Stop words (common words like 'the', 'is', etc.) are removed using NLTK’s stopwords list.
Feature Extraction:

The TF-IDF Vectorizer is used to convert the text data into numerical form. It creates a document-term matrix that captures the importance of each word in the text.
Model:

The Multinomial Naive Bayes (MNB) classifier is used for text classification. It works well for high-dimensional data such as text.


Requirements
To run the project, install the required libraries using:
  ```pip install -r requirements.txt```

## Usage:
To run this project use Jupyter Notebook or Google Colab.
OR 
You can run the following command in the terminal:
  ```python model.py```

## Evaluation
**Accuracy:** The accuracy score is the percentage of correct predictions (both spam and ham).
**Precision:** Precision is the ratio of correctly predicted spam messages to all predicted spam messages.
After training the model, the following results were obtained:

Accuracy: The model correctly classified 96.7% of all messages.
Precision: Of all the messages predicted to be spam, 99.35% were actually spam.

Confusion Matrix:
[[1195    1]
 [  45  152]]
