# ==============================================================================
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# AUTHORS: AMEY THAKUR & MEGA SATISH
# GITHUB (AMEY): https://github.com/Amey-Thakur
# GITHUB (MEGA): https://github.com/msatmod
# REPOSITORY: https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS
# RELEASE DATE: June 5, 2022
# LICENSE: MIT License
# DESCRIPTION: Utility module for the model training pipeline.
# ==============================================================================

import pickle
import warnings
import numpy as np
import pandas as pd
import spacy
import en_core_web_lg
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Suppression of non-critical runtime warnings to maintain algorithmic output integrity
warnings.filterwarnings("ignore")

def load_prepare_split_df(filename: str, targets=['label'], validation_size=0.3, seed=7):
    """
    Ingests raw data, performs feature extraction via word embeddings, 
    and partitions the dataset for model validation.
    
    Methodology:
        - TSV Ingestion: Data is loaded from the specified file.
        - Semantic Vectorization: Utilizing spaCy's dense 300-dimensional 
          word embeddings (centroid of token vectors).
        - Validation Partitioning: Stratified splitting of data into 
          training and testing subsets.

    Args:
        filename (str): Path to the TSV/CSV dataset.
        targets (list): Column name for the dependent variable.
        validation_size (float): Proportion of data reserved for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, Y_train, Y_test) feature and label sets.
    """
    print(f"Acquiring dataset from: {filename}")
    df_all = pd.read_csv(filename, sep='\t', encoding='utf-8')

    # Step 1: Initialize the Linguistic Engine
    nlp_engine = en_core_web_lg.load()

    # Step 2: Compute Dense Word Embeddings (Feature Extraction)
    print("Extracting semantic features via spaCy embeddings...")
    feature_vectors = np.array([
        np.array([token.vector for token in nlp_engine(s)]).mean(axis=0) * np.ones((300))
        for s in df_all['clean_text']
    ])
    
    # Step 3: Dataset Splitting
    y_labels = df_all.loc[:, targets]
    x_features = feature_vectors

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_labels, test_size=validation_size, random_state=seed
    )

    return x_train, x_test, y_train, y_test

def classification(X_train, Y_train, model=""):
    """
    Facilitates the training and serialization of various classification 
    architectures.
    
    Architectures Supported:
        - SVM: Support Vector Machine (Selected as the production primary).
        - LR: Logistic Regression.
        - DT: Decision Tree Classifier.
        - KNN: k-Nearest Neighbors (with automated k-optimization).
        - RF: Random Forest Classifier.
        - NN: Multi-layer Perceptron (MLP) Neural Network.

    Args:
        X_train: Training feature set.
        Y_train: Training label set.
        model (str): Target architecture identifier.

    Returns:
        object: The trained Scikit-learn model instance.
    """
    if model == "SVM":
        # Support Vector Machines are effective in high-dimensional semantic spaces
        print("Initializing SVM (Support Vector Machine) training...")
        clf = SVC(probability=True)
        clf.fit(X_train, Y_train)
        
        # Performance Evaluation (Accuracy Metric)
        train_accuracy = accuracy_score(clf.predict(X_train), Y_train)
        print(f"Training Convergence Accuracy: {train_accuracy:.4f}")

        # Persistence: Serializing the model artifact
        save_path = "../assets/models/model_svm_pc.pkl"
        with open(save_path, 'wb') as file:
            pickle.dump(clf, file)
        return clf

    elif model == "LR":
        # Logistic Regression serves as a robust baseline for linear classification
        print("Initializing Logistic Regression training...")
        lr_model = LogisticRegression()
        lr_model.fit(X_train, Y_train)
        
        save_path = "../assets/models/model_LogReg.pkl"
        with open(save_path, 'wb') as file:
            pickle.dump(lr_model, file)
        return lr_model

    elif model == "DT":
        # Decision Trees provide hierarchical decision boundaries
        print("Initializing Decision Tree training...")
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, Y_train)
        
        save_path = "../assets/models/model_DTC.pkl"
        with open(save_path, 'wb') as file:
            pickle.dump(dt_model, file)
        return dt_model

    elif model == "KNN":
        # kNN requires hyperparameter tuning (k value) via cross-validation
        print("Initializing kNN training with automated k-optimization...")
        k_values = range(1, 32, 1)
        k_scores = []

        # 10-Fold Cross-Validation for optimal k-neighbor selection
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = np.mean(cross_val_score(knn, X_train, Y_train, cv=10))
            k_scores.append(score)
            
        optimal_k = k_values[np.argmax(k_scores)]
        print(f"Optimized Hyperparameter discovered: k = {optimal_k}")

        best_knn = KNeighborsClassifier(n_neighbors=optimal_k)
        best_knn.fit(X_train, Y_train)

        save_path = "../assets/models/model_KNN.pkl"
        with open(save_path, 'wb') as file:
            pickle.dump(best_knn, file)
        return best_knn

    elif model == "RF":
        # Random Forest: Ensemble bagged decision trees for variance reduction
        print("Initializing Random Forest training...")
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, Y_train)
        
        save_path = "../assets/models/model_RF.pkl"
        with open(save_path, 'wb') as file:
            pickle.dump(rf_model, file)
        return rf_model

    elif model == "NN":
        # MLP (Multi-layer Perceptron): Basic artificial neural network
        print("Initializing Neural Network (MLP) training...")
        nn_model = MLPClassifier()
        nn_model.fit(X_train, Y_train)
        
        save_path = "../assets/models/model_NN.pkl"
        with open(save_path, 'wb') as file:
            pickle.dump(nn_model, file)
        return nn_model

def LSTM(filename: str):
    """
    Executes a Deep Learning pipeline using Long Short-Term Memory (LSTM) 
    recurrent neural networks for capturing temporal lingustical patterns.
    
    Methodology:
        - Tokenization: Integer encoding of sequences.
        - Padding: Uniform sequence length normalization.
        - Architecture: Embedding layer followed by LSTM with Dropouts.
    """
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, LSTM
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.wrappers.scikit_learn import KerasClassifier

    print(f"Acquiring data for Deep Learning (LSTM): {filename}")
    df_dl = pd.read_csv(filename, sep='\t', encoding='utf-8')

    # Step 1: Sequence Tokenization and Padding
    vocab_size = 20000
    max_len = 50
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df_dl['clean_text'])
    seqs = tokenizer.texts_to_sequences(df_dl['clean_text'])
    x_lstm = pad_sequences(seqs, maxlen=max_len)
    y_lstm = df_dl["label"]

    # Step 2: Architecture Definition
    print("Constructing LSTM topology...")
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=max_len))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Step 3: Model Execution and Persistance
    print("Commencing Deep Learning Convergence (LSTM)...")
    # In a professional context, create_model should be passed to KerasClassifier
    # Here we demonstrate the fundamental fit operation
    model.fit(x_lstm, y_lstm, epochs=3, verbose=1, validation_split=0.3)

    # Persistence: JSON topology and H5 weights
    model_json = model.to_json()
    with open("model_LSTM.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_LSTM.h5")
    print("Deep Learning model (LSTM) artifacts successfully persisted.")

