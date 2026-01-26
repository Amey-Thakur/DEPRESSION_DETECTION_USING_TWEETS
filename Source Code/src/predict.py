## Name: Milad Rezazadeh
## Description:
## Driver script for prediction.

###############################################################

## Import required libraries

import clean_utilities as CU
import argparse

## for data
import numpy as np
import pandas as pd

## Neglect the warnings!
import warnings
warnings.filterwarnings("ignore")

## for saving and loading model
import pickle

## for word embedding with Spacy
import spacy
import en_core_web_lg
###############################################################

## Initialize the parser
parser = argparse.ArgumentParser(
    description="Prediction"
)

## Add the first parameter (positional)
parser.add_argument('filename', help="The tweet you want to be examined")

## Add the second parameter (positional)
parser.add_argument('model', help="The type of model to be used")

## Parse the arguments
args = parser.parse_args()

## Conditioning the arguments
if args.filename is not None and args.model == "SVM":
    print('The file name is {}'.format(args.filename))
    with open(args.filename, 'r') as file:
        ## Retrieve the file content
        test_tweet = file.read()
        print("Tweet: ", test_tweet)
        ## Call tweets_cleaner function to clean the tweet
        print("cleaning...")
        clean_tweet = []
        clean_tweet.append(CU.tweets_cleaner(test_tweet))
        print("Clean tweet:", clean_tweet)
    print("word embedding ...")
    ## load English model of Spacy
    nlp = en_core_web_lg.load()
    ## word-embedding
    test = pd.np.array([pd.np.array([token.vector for token in nlp(s)]).mean(axis=0) * pd.np.ones((300)) \
                        for s in clean_tweet])

    ## Load the model
    SVM = \
        "/Users/milad/OneDrive - Dalhousie University/Depression_Detection/twitter_depression_detection/models/model_svm1.pkl"
    with open(SVM, 'rb') as file3:
        clf = pickle.load(file3)
        print("model==> ",clf)
    ## prediction
    labels_pred = clf.predict(test)
    result = labels_pred[0]
    if result == 1:
        print("Your tweet seems to be depressive")
    else:
        print("Your tweet is non-depressive")

else:print("Please provide your file and choose SVM as a model")


