## Name: Milad Rezazadeh
## Description:
## Utilities containing functions to train classification models and save them

###############################################################

## Import required libraries

## warnings
import warnings
warnings.filterwarnings("ignore")

## for data
import numpy as np
import pandas as pd

## Train-Test Split
from sklearn.model_selection import train_test_split

## Feature selection
from sklearn import feature_selection

## libraraies for classification
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

## for saving model
import pickle

## for word embedding with Spacy
import spacy
import en_core_web_lg

###############################################################

## function to takes the file_name as an input
## then returns the separated training and testing dataset

def load_prepare_split_df(filename, targets = ['label'], validation_size = 0.3, seed = 7):
    print("Gathering data from", filename)
    ## import to pandas dataframe
    df_all = pd.read_csv(filename, sep='\t', encoding='utf-8')

    ## load English model of Spacy
    nlp = en_core_web_lg.load()

    ## word-embedding
    all_vectors = pd.np.array([pd.np.array([token.vector for token in nlp(s)]).mean(axis=0) * pd.np.ones((300)) \
                               for s in df_all['clean_text']])
    # split out validation dataset for the end
    Y = df_all.loc[:, targets]
    X = all_vectors

    from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)

    ## return the training and testing features and labels (targets)
    return X_train, X_test, Y_train, Y_test

###############################################################

## Function to take three mandatory arguments and
## perform classification models
def classification(X_train,Y_train, model = ""):
    if model == "SVM": ## if classification is Support Vector Machine
        print("Building SVM model.")
        clf = SVC(probability=True)

        ## Full Training period
        print("Training ...")
        res = clf.fit(X_train, Y_train)
        train_result = accuracy_score(res.predict(X_train), Y_train)
        print("Train result ==> ", train_result)

        ## Save the model
        print("Saving the model: ")
        SVM = \
            "/Users/milad/OneDrive - Dalhousie University/Depression_Detection/twitter_depression_detection/models/model_svm_pc.pkl"
        with open(SVM, 'wb') as file:
            pickle.dump(clf, file)

        ## return the model
        return clf

    elif model == "LR": ## if the classification is Logistic Regression
        import sklearn.model_selection as skms
        print("Building Logistic Regression model.")
        LR = LogisticRegression()

        ## Full Training period
        print("Training ...")
        res = LR.fit(X_train, Y_train)
        train_result = accuracy_score(res.predict(X_train), Y_train)
        print("Train result ==> ", train_result)

        ## Save the model
        print("Saving the model: ")
        LogReg = ".../twitter_depression_detection/models/model_LogReg.pkl"
        with open(LogReg, 'wb') as file:
            pickle.dump(LR, file)

        ## return the model
        return LR


    elif model == "DT": ## if classification is Decision Tree
        print("Building Decision Tree model.")
        dtc = DecisionTreeClassifier()

        ## Full Training period
        print("Training ...")
        res = dtc.fit(X_train, Y_train)
        train_result = accuracy_score(res.predict(X_train), Y_train)
        print("Train result ==> ", train_result)

        ## Save the model
        print("Saving the model: ")
        DTC = ".../twitter_depression_detection/models/model_DTC.pkl"
        with open(DTC, 'wb') as file:
            pickle.dump(dtc, file)

        ## return the model
        return dtc

    elif model == "KNN": ## if classification is Decision Tree
        print("Building kNN model.")
        ## perform 10-fold cross-validation on a kNN model
        import sklearn.model_selection as skms
        import sklearn.neighbors as skn
        ## k values from 1 - 31 inclusive
        kvalues = range(1, 32, 1)
        scores = np.zeros(len(kvalues))

        ## Find optimal k value with cross-validation
        for i, k in enumerate(kvalues):
            kNN_model = skn.KNeighborsClassifier(k)
            scores[i] = np.mean(skms.cross_val_score(kNN_model, X_train, Y_train, cv=10))
        ## Report optimal value of k
        optimal_k = np.argmax(scores)
        print("Optimal value of k is ", optimal_k)

        ## best model based on optimal value for k
        best_kNN_model = skn.KNeighborsClassifier(optimal_k)
        ## Training based on the best model
        print("Training ...")
        best_kNN_model = best_kNN_model.fit(X_train, Y_train)

        ## Save the model
        print("Saving the model: ")
        KNN = ".../twitter_depression_detection/models/model_KNN.pkl"
        with open(KNN, 'wb') as file:
            pickle.dump(best_kNN_model, file)
        ## Return the best model
        return best_kNN_model

    elif model == "RF": ## if classification is Decision Tree
        print("Building Random Forest Classifiers.")
        rf = RandomForestClassifier()

        ## Full Training period
        print("Training ...")
        res = rf.fit(X_train, Y_train)
        train_result = accuracy_score(res.predict(X_train), Y_train)
        print("Train result ==> ", train_result)

        ## Save the model
        print("Saving the model: ")
        RF = ".../twitter_depression_detection/models/model_RF.pkl"
        with open(RF, 'wb') as file:
            pickle.dump(rf, file)

        ## return the model
        return rf

    elif model == "NN": ## if classification is Decision Tree
        print("Building Neural Network MLPClassifiers.")
        mlp = MLPClassifier()

        ## Full Training period
        print("Training ...")
        res = mlp.fit(X_train, Y_train)
        train_result = accuracy_score(res.predict(X_train), Y_train)
        print("Train result ==> ", train_result)

        ## Save the model
        print("Saving the model: ")
        NN = ".../twitter_depression_detection/models/model_NN.pkl"
        with open(NN, 'wb') as file:
            pickle.dump(mlp, file)

        ## return the model
        return mlp

###############################################################
def LSTM(filename):
    print("Gathering data from", filename)
    ## import to pandas dataframe
    df_all = pd.read_csv(filename, sep='\t', encoding='utf-8')

    ## load English model of Spacy
    nlp = en_core_web_lg.load()

    ### Create sequence
    vocabulary_size = 20000
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(df_all['clean_text'])
    sequences = tokenizer.texts_to_sequences(df_all['clean_text'])
    X_LSTM = pad_sequences(sequences, maxlen=50)
    nput_length = 50
    ## Split the data into train and test
    Y_LSTM = df_all["label"]
    X_train_LSTM, X_test_LSTM, Y_train_LSTM, Y_test_LSTM = train_test_split(X_LSTM, Y_LSTM, test_size=validation_size, random_state=seed)

    from keras.wrappers.scikit_learn import KerasClassifier
    print("Building LSTM model.")
    model = Sequential()
    model.add(Embedding(20000, 300, input_length=50))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    ## Training the model
    print("Training LSTM ...")
    model_LSTM = KerasClassifier(build_fn=create_model, epochs=3, verbose=1, validation_split=0.4)
    model_LSTM.fit(X_train_LSTM, Y_train_LSTM)

    # serialize model to JSON
    model_json = model_LSTM.to_json()
    with open("model_LSTM.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_LSTM.h5")
    print("Saved model to disk")

