# ==============================================================================
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# AUTHORS: AMEY THAKUR & MEGA SATISH
# GITHUB (AMEY): https://github.com/Amey-Thakur
# GITHUB (MEGA): https://github.com/Mega-Satish
# REPOSITORY: https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS
# RELEASE DATE: June 5, 2022
# LICENSE: MIT License
# DESCRIPTION: Command-line interface (CLI) script for the linguistic 
#              preprocessing of raw tweet data. Utilizes advanced NLP 
#              techniques for data normalization and sanitation.
# ==============================================================================

import argparse
import warnings
import clean_utilities as CU

# Suppression of non-critical runtime warnings to ensure output clarity
warnings.filterwarnings("ignore")

def main():
    """
    Primary execution routine for the tweet cleaning utility.
    
    This script facilitates the transformation of raw unstructured text 
    into a standardized format, essential for downstream machine learning 
    inference and training.
    """
    # Configuration of the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Twitter Depression Detection: Linguistic Preprocessing Utility"
    )

    # Definition of the mandatory positional argument for input file path
    parser.add_argument(
        'filename', 
        help="Path to the raw text file containing the tweet to be sanitized"
    )

    # Parsing and validation of terminal arguments
    args = parser.parse_args()

    # Conditional logic to verify input availability before processing
    if args.filename is not None:
        print(f"Targeting file for preprocessing: {args.filename}")
        
        try:
            # Atomic read operation for the target text file
            with open(args.filename, 'r', encoding='utf-8') as file:
                raw_tweet = file.read()
                
                # Invocation of the granular cleaning pipeline
                # Methodology includes contraction expansion, tokenization, and lemmatization
                print("Linguistic cleaning in progress...")
                sanitized_tweet = CU.tweets_cleaner(raw_tweet)
                
                # Persisting the sanitized result to local storage
                with open('clean_tweet.txt', 'w', encoding='utf-8') as output_file:
                    print("Sanitization complete. Persistence target: clean_tweet.txt")
                    output_file.write(sanitized_tweet)
                    
        except FileNotFoundError:
            print(f"Error: The specified file '{args.filename}' was not discovered.")
        except Exception as e:
            print(f"An unexpected analytical error occurred: {e}")
            
    else:
        print("Required input: Please specify a valid filename as a positional argument.")

if __name__ == '__main__':
    main()



