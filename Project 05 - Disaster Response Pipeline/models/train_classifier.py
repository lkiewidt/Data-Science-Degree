import sys
import numpy as np
import pandas as pd
import sqlalchemy as db
import re
import nltk
from nltk.corpus import stopwords
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# download stopwords
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Loads data from database and returns features, labels, and category names.
    
    Inputs
    ------
    database_filepath : str
        path to database
        
    Returns
    -------
    X : ndarray
        features
        
    Y : ndarray
        labels
    
    category_names : str
        list with category names
    """
    
    # load data from database
    engine = db.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('cleaned_messages', engine)

    # assign features and labels
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    
    # get category names
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns.to_list()
    
    return X, Y, category_names


def tokenize(text):
    """
    Processes the provided text input:
    - normalize
    - tokenize
    - remove stopwords
    - lemmatization and stemming
    
    Inputs
    ------
    text : str
        text to process
    
    Returns
    -------
    stemmed : list
        list of processed words in the original text
    """
    
    # normalize to lower case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize message
    tokens = nltk.tokenize.word_tokenize(text)
    
    # remove stopwors (English)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    # lemmatize and stem words
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()
    
    lemmed = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = [stemmer.stem(lemma) for lemma in lemmed]
    
    return stemmed


def build_model():
    """
    Build machine learning model to classify messages.
    
    Returns
    -------
    model : sklearn classification pipeline
        untrained model
    """
    
    # construct ML pipeline
    steps = [('count', CountVectorizer(tokenizer=tokenize)),
             ('tfidf', TfidfTransformer()),
             ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
            ]
    
    pipeline = Pipeline(steps=steps)
    
    # run grid search to optimize model parameters
    # RandomForest parameters
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 10, 50, 100]
    }
    
    # enable refitting with best set of parameters
    model = GridSearchCV(pipeline, param_grid=parameters, refit=True)
    
    return model


def report_model_scores(tests, predictions, classes):
    """
    Report precision, recall, and f1 scores for each class
    
    Parameters
    ----------
    tests : ndarray
        true labels
        
    predictions : ndarray
        predicted labels
        
    classes : list
        classes
        
    Returns
    -------
    scores : ndarray
        array of calculated scores (class in row, scores in columns: precision, recall, f1)
    """
    # get number of classes and initialize scores array
    n_classes = len(classes)
    scores = np.zeros((n_classes, 3))
    
    # calculate precision, recall, and f1 for each class
    for i in range(n_classes):
        scores[i, 0] = precision_score(tests[:, i], predictions[:, i], average='weighted')
        scores[i, 1] = recall_score(tests[:, i], predictions[:, i], average='weighted')
        scores[i, 2] = f1_score(tests[:, i], predictions[:, i], average='weighted')
    
    # sort scores in ascending order by f1 score
    idx = np.argsort(scores, axis=0)[:, 0]
    
    # print scores for each class in ascending order
    for i in range(n_classes):
        print("{0:25s}{1:.2f}\t{2:.2f}\t{3:.2f}".format(classes[idx[i]], scores[idx[i], 0],
                                                        scores[idx[i], 1], scores[idx[i], 2]))
    
    # calculate and print average scores
    averages = np.average(scores, axis=0)
    print("\n{0:25s}{1:.2f}\t{2:.2f}\t{3:.2f}".format("AVERAGES", averages[0], averages[1], averages[2]))
    
    return scores


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predicts categories on test data and reports precision, recall, and f1
    for each category.
    
    Inputs
    ------
    model : sklearn classification pipeline
        trained classifier model

    X_test : ndarray
        features of test dataset
    
    Y_test: ndarray
        labels of test dataset
        
    category_names : list (str)
        list of category names
    """
    # predict classes on test set
    Y_pred = model.predict(X_test)
    
    # calculate and print scores for each category
    scores = report_model_scores(Y_test, Y_pred, category_names)
    
    return 0


def save_model(model, model_filepath):
    """
    Save trained classifier model.
    
    Inputs
    ------
    model : sklearn classifier object
        trained model to be saved
        
    model_filepath : str
        filepath to save model
    """
    
    # save trained model with pickle
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()

    return 0


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training/optimizing model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()