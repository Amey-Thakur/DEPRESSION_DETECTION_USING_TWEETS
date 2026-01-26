## Name: Milad Rezazadeh
## Description:
## Driver script for classification models.

###############################################################

## Import required libraries

import train_utilities as TU
import argparse
## Neglect the warnings!
import warnings
warnings.filterwarnings("ignore")

################################################################

## Initialize the parser
parser = argparse.ArgumentParser(
    description="Training"
)

## Add the first parameter (positional)
parser.add_argument('filename', help="The URL of the data set to be examined")

## Add the second parameter (positional)
parser.add_argument('model', help="The type of model to be used")

## Parse the arguments
args = parser.parse_args()

## Conditioning the arguments

if args.model == "DT": ## Desicion Tree model
    ## load and prepare the dataset
    X_train, X_test, Y_train, Y_test = TU.load_prepare_split_df(args.filename)

    ## Create the model and train it
    pretrained_model = TU.classification(X_train = X_train,Y_train = Y_train, model="DT")

elif args.model == "LR":
    ## load and prepare the dataset
    X_train, X_test, Y_train, Y_test = TU.load_prepare_split_df(args.filename)

    ## Create the model and train it
    pretrained_model = TU.classification(X_train, Y_train, model="LR")

elif args.model == "kNN":
    ## load and prepare the dataset
    X_train, X_test, Y_train, Y_test = TU.load_prepare_split_df(args.filename)

    ## Create the model and train it
    pretrained_model = TU.classification(X_train = X_train,Y_train = Y_train, model="KNN")

elif args.model == "SVM":
    ## load and prepare the dataset
    X_train, X_test, Y_train, Y_test = TU.load_prepare_split_df(args.filename)

    ## Create the model and train it
    pretrained_model = TU.classification(X_train = X_train,Y_train = Y_train, model="SVM")

elif args.model == "RF":
    ## load and prepare the dataset
    X_train, X_test, Y_train, Y_test = TU.load_prepare_split_df(args.filename)

    ## Create the model and train it
    pretrained_model = TU.classification(X_train = X_train,Y_train = Y_train, model="RF")

elif args.model == "NN":
    ## load and prepare the dataset
    X_train, X_test, Y_train, Y_test = TU.load_prepare_split_df(args.filename)

    ## Create the model and train it
    pretrained_model = TU.classification(X_train = X_train,Y_train = Y_train, model="NN")

elif args.model == "LSTM":
    ## load and prepare the dataset
    TU.LSTM(args.filename)