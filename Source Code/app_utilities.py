# ==============================================================================
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# AUTHORS: AMEY THAKUR & MEGA SATISH
# GITHUB (AMEY): https://github.com/Amey-Thakur
# GITHUB (MEGA): https://github.com/msatmod
# REPOSITORY: https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS
# RELEASE DATE: June 5, 2022
# LICENSE: MIT License
# DESCRIPTION: Utility module for tweet analysis predictions.
# ==============================================================================

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import spacy
import en_core_web_lg
# Configure sys.path to permit localized module discovery within the core directory
sys.path.append('./core')

import clean_utilities as CU

# Suppression of non-critical runtime warnings to maintain a clean console log
warnings.filterwarnings("ignore")

def tweet_prediction(tweet: str) -> int:
    """
    Takes a tweet and returns whether it's classified as depressive (1) or not (0).
    
    The process:
        1. Clean the text using our utility module.
        2. Convert text to numbers using spaCy.
        3. Use the trained SVM model to make a prediction.
    Args:
        tweet (str): The tweet text from the user.

    Returns:
        int: 1 for Depressive, 0 for Non-depressive.
    """
    # Step 1: Clean the text
    processed_tweet = tweet
    cleaned_input = []
    cleaned_input.append(CU.tweets_cleaner(processed_tweet))
    
    # Step 2: Convert text to numbers using spaCy
    nlp_engine = en_core_web_lg.load()
    
    # Step 3: Compute centroid word embeddings
    # We calculate the mean vector of all tokens to represent the tweet's semantic context
    semantic_vectors = np.array([
        np.array([token.vector for token in nlp_engine(s)]).mean(axis=0) * np.ones((300))
        for s in cleaned_input
    ])
    
    # Step 4: Load the pre-trained Support Vector Machine (SVM) model artifact
    # The SVM was selected for its robust performance in high-dimensional text classification
    model_path = "./assets/models/model_svm1.pkl"
    with open(model_path, 'rb') as model_file:
        classifier = pickle.load(model_file)
    
    # Step 5: Perform binary classification
    prediction_result = classifier.predict(semantic_vectors)
    
    return int(prediction_result[0])
