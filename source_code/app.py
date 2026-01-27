# ==============================================================================
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# AUTHORS: AMEY THAKUR & MEGA SATISH
# GITHUB (AMEY): https://github.com/Amey-Thakur
# GITHUB (MEGA): https://github.com/msatmod
# REPOSITORY: https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS
# RELEASE DATE: June 5, 2022
# LICENSE: MIT License
# DESCRIPTION: Flask application entry point for the tweet analysis project.
# ==============================================================================

#!/usr/bin/env python3

import pickle
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
import app_utilities

import nltk
# Download necessary NLTK data for the cleaning pipeline
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Initialize the Flask application
# Flask-Bootstrap is utilized for enhanced UI styling consistency
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    """Renders the landing page for tweet input."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission and displays the prediction result.
        
    Returns:
        Rendered result HTML with the model's prediction outcome.
    """
    if request.method == 'POST':
        # Retrieve the tweet content submitted via the web interface
        tweet = request.form["tweet"]
        input_data = [tweet]
        
        # Invoke the backend prediction utility to classify the tweet's sentiment
        # The engine utilizes an SVM classifier with spaCy word embeddings
        my_prediction = app_utilities.tweet_prediction(str(input_data))
        
        return render_template("result.html", prediction=my_prediction, name=tweet)

@app.errorhandler(404)
def page_not_found(e):
    """
    Custom 404 error handler.
    Renders the personalized 404 page when a resource is not found.
    """
    return render_template('404.html'), 404

# Entry point for the Flask development server
if __name__ == '__main__':
    # Execution on port 7860 as required for Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)
