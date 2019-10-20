import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterMessages.db')
df = pd.read_sql_table('cleaned_messages', engine)

# load model
model = joblib.load("../models/message_classifier.pkl")

# assign features and labels
X = df['message'].values
Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values

# split into test and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=True, random_state=42)

# make predictions on test data
Y_predict = model.predict(X_test)

# calculate f1 score for each class
f1_scores = np.zeros(Y_test.shape[1])

for i in range(f1_scores.size):
    f1_scores[i] = f1_score(Y_test[:, i], Y_predict[:, i], average='macro')
    

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)
    # get class names
    class_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns.to_list()
    class_counts = df[class_names].sum(axis=0).values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=class_names,
                    y=f1_scores
                )
            ],

            'layout': {
                'title': 'f1 score for each class',
                'yaxis': {
                    'title': "f1 score"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                Bar(
                    x=class_names,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classes',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()