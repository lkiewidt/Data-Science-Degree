# P-05: Desaster Response Pipeline

## Introduction
In the Desaster Response project, text messages send during disaster situations were analyzed and classified into several different categories (e.g. fire, flood, storm, etc.) to allow quick and effective deployment of support. Therefore, ETL (extract-transform-load) and ML (machine learning) pipelines were implemented to enable automated and straightfoward processing of datasets, and training and testing of the classifier. The trained classifier can be used in a web-based app to classify incoming messages.

## Approach
First, an ETL pipeline was established to automatically extract, clean, and store the messages and categories into a database. The different categories are one-hot encoded. Afterwards, a ML pipeline was implemented to automatically train a classifier, optimize its hyper-parameters, and evaluate its performance. Prior to training the classifier, the text messages were normalized (lowercase, punctuation removed) and tokenized into individual words. Afterwards, stopwords were removed and the remaining words were lemmatized and stemmed. Finally, TfIdf (term-frequency, inverse-document-frequency) was used to transform the relevant information in the messages into numerical data.

## Python Libraries
The following Python libraries are use in the project:
* python = 3.7.3
* numpy = 1.16.4
* pandas = 0.24.2
* matplotlib = 3.1.0
* seaborn = 0.9.0

## Usage
Repository files:
`data/ETL Pipeline Preparation.ipynb`: Jupyter notebook to develop the ETL pipeline
`data/process_data.py`: Python script to run the ETL pipeline
`model/ML Pipeline Preparation.ipynb`: Jupyter notebook to develop the ML pipeline
`model/train_classifier`: Python script to run the ML pipeline
`app/run.py`: Python script to start the web-based app

To process and clean new datasets, run

`python process_data.py 'path/to/messages.csv' 'path/to/categories.csv' 'path/to/database.db'`

The function will load and the data from the to datasets (csv-files), clean the data, and store it into a database.

To train the classifier on a new dataset and to save the trained model in a .pkl file run

`python train_classifier.py 'path/to/database.db' 'path/to/save/model.pkl`

Finally, to run the web-based app run

`python run.py`

in the `app` directory and go to `localhost:3001` in your browser. You will see stats of the training dataset, a input slit for new messages, and a classify button to classify the message in the input slit.