#!flask/bin/python
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
import pickle
import app_utilities

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Collect the input and predict the outcome
    Returns:
        Results.html with prediction
    """
    if request.method == 'POST':
        # get input statement
        tweet = request.form["tweet"]
        data = [tweet]
        my_prediction = app_utilities.tweet_prediction(str(data))
    return render_template("result.html", prediction=my_prediction, name=tweet)


if __name__ == '__main__':
    app.run(debug=True)
