# ==============================================================================
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# AUTHORS: AMEY THAKUR & MEGA SATISH
# GITHUB (AMEY): https://github.com/Amey-Thakur
# GITHUB (MEGA): https://github.com/msatmod
# REPOSITORY: https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS
# RELEASE DATE: June 5, 2022
# LICENSE: MIT License
# DESCRIPTION: Script for training machine learning models for tweet analysis.
# ==============================================================================

import argparse
import warnings
import train_utilities as TU

# Suppression of non-critical runtime warnings to ensure output clarity during training
warnings.filterwarnings("ignore")

def main():
    """
    Primary execution routine for the model training utility.
    
    This script facilitates the training of various machine learning 
    architectures by providing a standardized interface for:
        1. Dataset Ingestion: Loading and splitting training data.
        2. Hyperparameter Configuration: Setting up model-specific parameters.
        3. Algorithmic Training: Executing the training process via train_utilities.
        4. Model Serialization: Persisting the resulting model for future inference.
    """
    # Initialize the CLI argument parser
    parser = argparse.ArgumentParser(
        description="Twitter Depression Detection: Model Training Utility"
    )

    # Positional argument for the training dataset path (CSV format)
    parser.add_argument(
        'filename', 
        help="Path to the training dataset (TSV/CSV format with 'label' and 'clean_text')"
    )

    # Positional argument for the classification model architecture
    # Supported: 'DT', 'LR', 'kNN', 'SVM', 'RF', 'NN', 'LSTM'
    parser.add_argument(
        'model', 
        help="Target model architecture for training"
    )

    # Execution of the parsing logic
    args = parser.parse_args()

    # Deployment of the selected training pipeline based on the 'model' parameter
    model_type = args.model
    dataset_path = args.filename

    # Pipeline selection logic
    if model_type in ["DT", "LR", "kNN", "SVM", "RF", "NN"]:
        # Logic for standardized Scikit-learn architectures
        print(f"Initializing {model_type} training pipeline...")
        
        # Step 1: Data Acquisition and Validation Splitting
        X_train, X_test, Y_train, Y_test = TU.load_prepare_split_df(dataset_path)

        # Step 2: Algorithmic Training and Parameter Optimization
        # The 'classification' method handles instantiation and fitting
        trained_model = TU.classification(X_train=X_train, Y_train=Y_train, model=model_type)
        
        print(f"Training for {model_type} successful.")

    elif model_type == "LSTM":
        # Specialized logic for Long Short-Term Memory (LSTM) Neural Networks
        # LSTMs are utilized here to capture long-range temporal dependencies in text
        print("Initializing LSTM deep learning pipeline...")
        TU.LSTM(dataset_path)
        
    else:
        print(f"Error: Model architecture '{model_type}' is not currently recognized.")
        print("Supported architectures: DT, LR, kNN, SVM, RF, NN, LSTM")

if __name__ == '__main__':
    main()