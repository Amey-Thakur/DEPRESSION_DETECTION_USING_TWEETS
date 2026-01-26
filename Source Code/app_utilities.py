# ==============================================================================
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# AUTHORS: AMEY THAKUR & MEGA SATISH
# GITHUB (AMEY): https://github.com/Amey-Thakur
# GITHUB (MEGA): https://github.com/Mega-Satish
# REPOSITORY: https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS
# RELEASE DATE: June 5, 2022
# LICENSE: MIT License
# DESCRIPTION: Backend utility module providing core sentiment classification 
#              logic for the web application, incorporating spaCy embeddings 
#              and a pre-trained SVM model.
# ==============================================================================

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import spacy
import en_core_web_lg
import clean_utilities as CU

# Configure sys.path to permit localized module discovery within the core directory
sys.path.append('./core')

# Suppression of non-critical runtime warnings to maintain a clean console log
warnings.filterwarnings("ignore")

def tweet_prediction(tweet: str) -> int:
    """
    Executes the analytical pipeline for a single tweet string to determine 
    its depressive characteristic.
    
    The pipeline consists of:
        1. Linguistic Cleaning: Utilizing the custom clean_utilities module.
        2. Vectorization: Applying spaCy's 'en_core_web_lg' model for dense 
           word embeddings (300-dimensional vectors).
        3. Inference: Processing the resulting vector through a pre-trained 
           Support Vector Machine (SVM) classifier.

    Args:
        tweet (str): The raw text input captured from the user interface.

    Returns:
        int: Binary classification result (1 for Depressive, 0 for Non-depressive).
    """
    # Step 1: Execute linguistic preprocessing
    processed_tweet = tweet
    cleaned_input = []
    cleaned_input.append(CU.tweets_cleaner(processed_tweet))
    
    # Step 2: Load the high-fidelity English transformer model
    nlp_engine = en_core_web_lg.load()
    
    # Step 3: Compute centroid word embeddings
    # We calculate the mean vector of all tokens to represent the tweet's semantic context
    semantic_vectors = pd.np.array([
        pd.np.array([token.vector for token in nlp_engine(s)]).mean(axis=0) * pd.np.ones((300))
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
