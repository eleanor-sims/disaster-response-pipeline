# disaster-response-pipeline

## Table of Contents
1. [Introduction](https://github.com/louisteo9/udacity-disaster-response-pipeline#introduction)
2. [File Descriptions](https://github.com/louisteo9/udacity-disaster-response-pipeline#file-descriptions)
3. [Installation](https://github.com/louisteo9/udacity-disaster-response-pipeline#installation)
4. [Instructions](https://github.com/louisteo9/udacity-disaster-response-pipeline#instructions)
5. [Acknowledgements](https://github.com/louisteo9/udacity-disaster-response-pipeline#acknowledgements)

## Introduction
The Disaster Response Pipeline is part of the Udacity Data Scientist Nanodegree Program. The data is provided by [Appen](https://www.appen.com/).

Pre-labelled disaster messages are used to train a model that can categorize new messages received in real time during actual disasters, so that these can be sent to appropriate disaster response agencies.

The project also includes a web application where visualisations of the training data are displayed and new messages can be input and classified based on the trained model.

## File Descriptions
The code for this project is organised as follows:
### Folder: app
**run.py** - python script needed to run web app.<br/>
Folder: templates - html files (go.html & master.html) required for the web app.

### Folder: data
**disaster_messages.csv** - real messages sent during disaster events (provided by Appen)<br/>
**disaster_categories.csv** - corresponding categories of the messages<br/>
**process_data.py** - ETL pipeline used to load, clean, extract feature and store data in SQLite database<br/>
**ETL Pipeline Preparation.ipynb** - Jupyter Notebook used for analysis and to prepare ETL pipeline<br/>
**DisasterResponse.db** - cleaned data stored in table df_clean in SQlite database

### Folder: models
**train_classifier.py** - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file .<br/>
**ML Pipeline Preparation.ipynb** - Jupyter Notebook used for analysis and to prepare ML pipeline

## Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.

## Instructions
1. Navigate to the project root directory and run the following commands to establish the database and model:

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Navigate to the app directory and run the following command:
    `python run.py`

3. Go to http://0.0.0.0:3001/ to view the web app

## Licensing, Authors, Acknowledgements
* Credit to [Udacity](https://www.udacity.com/) for the course materials and script templates.
* Credit to [Appen](https://www.appen.com) for providing the datasets used to create the model and visualisations.

