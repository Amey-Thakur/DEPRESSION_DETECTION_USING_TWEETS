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
# Global initialization of heavy resources to optimize runtime performance
# Loading these once at startup eliminates significant latency during individual requests

# 1. Load spaCy NLP engine
try:
    nlp_engine = en_core_web_lg.load()
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    sys.exit(1)

# 2. Load pre-trained SVM Classifier
model_path = "./assets/models/model_svm1.pkl"
try:
    with open(model_path, 'rb') as model_file:
        classifier = pickle.load(model_file)
except Exception as e:
    print(f"Error loading SVM model from {model_path}: {e}")
    sys.exit(1)

def tweet_prediction(tweet: str) -> int:
    """
    Takes a tweet and returns whether it's classified as depressive (1) or not (0).
    
    The process:
        1. Clean the text using our utility module.
        2. Convert text to numbers using the pre-loaded spaCy engine.
        3. Use the pre-loaded SVM model to make a prediction.
    Args:
        tweet (str): The tweet text from the user.

    Returns:
        int: 1 for Depressive, 0 for Non-depressive.
    """
    # Step 1: Clean the text
    cleaned_text = CU.tweets_cleaner(tweet)
    
    # Step 2: Compute centroid word embeddings
    # We calculate the mean vector of all tokens to represent the tweet's semantic context
    # Note: Global 'nlp_engine' is used here, avoiding reload overhead
    vector = np.array([token.vector for token in nlp_engine(cleaned_text)]).mean(axis=0) * np.ones((300))
    semantic_vectors = np.array([vector])
    
    # Step 3: Perform binary classification
    # Note: Global 'classifier' is used here
    prediction_result = classifier.predict(semantic_vectors)
    
    return int(prediction_result[0])
