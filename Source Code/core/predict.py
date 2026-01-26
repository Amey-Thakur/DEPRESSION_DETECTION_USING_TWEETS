# ==============================================================================
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# AUTHORS: AMEY THAKUR & MEGA SATISH
# GITHUB (AMEY): https://github.com/Amey-Thakur
# GITHUB (MEGA): https://github.com/Mega-Satish
# REPOSITORY: https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS
# RELEASE DATE: June 5, 2022
# LICENSE: MIT License
# DESCRIPTION: Command-line interface (CLI) driver script for performing 
#              sentiment inference on individual tweets using a pre-trained 
#              Support Vector Machine (SVM) and spaCy word embeddings.
# ==============================================================================

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import spacy
import en_core_web_lg
import clean_utilities as CU

# Suppression of non-critical runtime warnings to maintain output integrity
warnings.filterwarnings("ignore")

def main():
    """
    Main entry point for the prediction utility.
    
    This script encapsulates the end-to-end inference pipeline:
        1. Argument Parsing: Captures input text file and model selection.
        2. Text Preprocessing: Normalization via clean_utilities.
        3. Feature Extraction: Generating centroid embeddings via spaCy.
        4. Classification: Binary sentiment analysis via pre-trained SVM.
    """
    # Initialize the CLI argument parser with a descriptive header
    parser = argparse.ArgumentParser(
        description="Twitter Depression Detection: Machine Learning Inference Utility"
    )

    # Positional argument for the target tweet content (text file)
    parser.add_argument(
        'filename', 
        help="Path to the text file containing the tweet for classification"
    )

    # Positional argument for the classification model type
    parser.add_argument(
        'model', 
        help="Target model architecture (currently optimized for 'SVM')"
    )

    # Execution of the parsing logic
    args = parser.parse_args()

    # Pipeline validation: Ensuring input availability and model compatibility
    if args.filename is not None and args.model == "SVM":
        print(f"Loading input source: {args.filename}")
        
        try:
            # Step 1: Data Acquisition
            with open(args.filename, 'r', encoding='utf-8') as file:
                raw_test_tweet = file.read()
                print(f"Captured Content: \"{raw_test_tweet}\"")
                
                # Step 2: Linguistic Preprocessing
                # Normalizes raw discourse into a tokenizable semantic format
                print("Executing linguistic cleaning pipeline...")
                cleaned_input = [CU.tweets_cleaner(raw_test_tweet)]
                print(f"Normalized Form: {cleaned_input}")

            # Step 3: Feature Space Transformation
            # Utilizing dense word embeddings (spaCy 'en_core_web_lg' model)
            print("Transforming text to 300-dimensional semantic vectors...")
            nlp_engine = en_core_web_lg.load()
            
            # Generating the centroid vector representing the tweet's linguistic context
            semantic_features = pd.np.array([
                pd.np.array([token.vector for token in nlp_engine(s)]).mean(axis=0) * pd.np.ones((300))
                for s in cleaned_input
            ])

            # Step 4: Model Artifact Loading
            # Loading the serialized SVM classifier from the assets directory
            model_artifact_path = "../assets/models/model_svm1.pkl"
            with open(model_artifact_path, 'rb') as model_file:
                classifier = pickle.load(model_file)
                
            # Step 5: Algorithmic Inference
            # The SVM determines the classification boundary for the semantic vector
            print("Performing binary classification...")
            prediction_bin = classifier.predict(semantic_features)
            
            # Step 6: Result Interpretation and User Communication
            is_depressive = prediction_bin[0]
            if is_depressive == 1:
                print("\n>>> CLASSIFICATION RESULT: The analyzed content exhibits depressive characteristics.")
            else:
                print("\n>>> CLASSIFICATION RESULT: The analyzed content is classified as non-depressive.")

        except FileNotFoundError:
            print(f"Error: The input file {args.filename} could not be located.")
        except Exception as e:
            print(f"An error occurred during the inference process: {e}")

    else:
        print("Usage Error: Please provide an input file and specify 'SVM' as the target model.")

if __name__ == '__main__':
    main()


