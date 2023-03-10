import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Function to load the data contained in the messages and categories csv files

        Inputs: messages and categories filepaths
        Return: merged dataframe df'''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id', how='left')

    return df


def clean_data(df):
    '''Function to clean the merged dataframe created via load_data(messages_filepath, categories_filepath).
        Extracts the categories from df and splits into multiple columns with binary row entries.
        Drops any rows where the category entries are not 0 or 1
        Drops any columns where all entries are the same (i.e. all 0 or 1)
        Drops any duplicate rows

        Input: dataframe created via load_data(messages_filepath, categories_filepath) df
        Returns: cleaned dataframe df'''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2]).values

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string
    for col in category_colnames:
        # set each value to be the last character of the string
        categories[col] = categories[col].astype(str).apply(lambda x: x[-1:]).values
        # convert column from string to numeric
        categories[col] = categories[col].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove rows where any of the categorical columns are not equal to 0 or 1
    for col in list(category_colnames):
        df = df.drop(df[(df[col] != 0) & (df[col] != 1)].index)

    # Remove columns where all target values are equivalent
    for col in list(category_colnames):
        if len(set(df[col])) == 1:
            df = df.drop(columns=col)

    # Drop duplicate rows
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''Function to save the cleaned data to an sql database

        Inputs: cleaned dataframe df, file name string database_filename
        Returns: None'''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    '''Function to run the main ETL program'''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()