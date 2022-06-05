## Name: Milad Rezazadeh
## Description:
## Script to clean tweets and save them

###############################################################

## Import clean_utilities.py

import clean_utilities as CU
import argparse

## Neglect the warnings!
import warnings
warnings.filterwarnings("ignore")

###############################################################

## Initialize the parser
parser = argparse.ArgumentParser(
    description="Cleaning"
)

## Add the first parameter (positional)
parser.add_argument('filename', help="The name of the file to be cleaned")

## Parse the arguments
args = parser.parse_args()

## Conditioning the arguments
if args.filename is not None:
    print('The file name is {}'.format(args.filename))
    with open(args.filename, 'r') as file:
        ## Retrieve the file content
        tweet = file.read()
        ## Call tweets_cleaner function to clean the tweet
        print("cleaning...")
        clean_tweet = CU.tweets_cleaner(tweet)
        #print("Saving the tweet in clean_tweet.txt")
        with open('clean_tweet.txt', 'w') as file2:
            print("Saving the clean tweet in clean_tweet.txt")
            file2.write(clean_tweet)
else: print("Please specify the filename")



