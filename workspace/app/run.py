import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine



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
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df_clean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/master')
def master():
    # extract data needed for visuals

    # TODO: Below is an example - modify to extract data for your own visuals
    # data for first graph
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # data for second graph
    category_names = [col for col in df.columns if col not in ['id', 'message', 'original', 'genre']]
    category_counts = df[category_names].sum().to_frame().rename(columns={0: 1}).sort_index()[1]

    # data for third graph
    df_cat = df[category_names]
    df_corr = df_cat.corr().values.tolist()

    # create graphs list
    graphs = [
        {
            "data": [
                {
                    "type": "bar",
                    "x": genre_names,
                    "y": genre_counts

                }
            ],
            "layout": {
                "title": "Distribution of Message Genres"
            }
        },
        {
            "data": [
                {
                    "type": "bar",
                    "x": category_names,
                    "y": category_counts

                }
            ],
            "layout": {
                "title": "Distribution of Message Categories"
            }
        },
        {
            "data": [
                {
                    "type": "heatmap",
                    "x": category_names,
                    "y": category_names,
                    "z": df_corr,
                    "colorscale": "purples_r",
                    "labels": category_names

                }
            ],
            "layout": {
                "title": "Heatmap of correlation between categories"
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()