## Import requried libraries
import sys
# sys.path is a list of absolute path strings
sys.path.append('../twitter_depression_detection/src')

import clean_utilities as CU
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


def tweet_prediction(tweet):
    test_tweet = tweet
    clean_tweet = []
    clean_tweet.append(CU.tweets_cleaner(test_tweet))
    ## load English model of Spacy
    nlp = en_core_web_lg.load()
    ## word-embedding
    test = pd.np.array([pd.np.array([token.vector for token in nlp(s)]).mean(axis=0) * pd.np.ones((300)) \
                        for s in clean_tweet])
    ## Load the model
    SVM = "../twitter_depression_detection/models/model_svm1.pkl"
    with open(SVM, 'rb') as file3:
        clf = pickle.load(file3)

    ## prediction
    labels_pred = clf.predict(test)

    # if labels_pred[0] == 1:
    #     pred = "Depressive"
    # else:
    #     pred = "Non-depressive"

    return labels_pred[0]
