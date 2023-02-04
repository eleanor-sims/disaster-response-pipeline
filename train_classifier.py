import sys
# download necessary NLTK data
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords'])

# import libraries
import numpy as np
import pandas as pd

import sqlite3
from sqlalchemy import create_engine

import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM df_clean', conn)

    target_col_list = [col for col in df.columns if col not in ['id', 'message', 'original', 'genre']]

    X = df['message']
    y = df[target_col_list]

    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    # define url regex to replace these with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # replace any urls with 'urlplaceholder'
    text = re.sub(url_regex, "urlplaceholder", text)

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # convert all to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Lemmatize
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC(loss='hinge')))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()