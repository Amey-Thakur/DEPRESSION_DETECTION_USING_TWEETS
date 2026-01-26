# ==============================================================================
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# AUTHORS: AMEY THAKUR & MEGA SATISH
# GITHUB (AMEY): https://github.com/Amey-Thakur
# GITHUB (MEGA): https://github.com/Mega-Satish
# REPOSITORY: https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS
# RELEASE DATE: June 5, 2022
# LICENSE: MIT License
# DESCRIPTION: Flask-based web application entry point for the Twitter 
#              Depression Detection system. This script handles routing, 
#              input collection, and serves the prediction results.
# ==============================================================================

#!/usr/bin/env python3

import pickle
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
import app_utilities

# Initialize the Flask application
# Flask-Bootstrap is utilized for enhanced UI styling consistency
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    """
    Renders the primary landing page of the application.
    
    Returns:
        The refined index1.html template containing the tweet input form.
    """
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Facilitates the inference pipeline by capturing user input, 
    invoking the prediction logic, and rendering the results.
    
    Process:
        1. Captures the 'tweet' string from the POST request form.
        2. Routes the input to the optimized app_utilities prediction engine.
        3. Returns the result.html template populated with the classification outcome.
        
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

# Entry point for the Flask development server
if __name__ == '__main__':
    # Execution with debugging enabled for rapid developmental iteration
    app.run(debug=True)
