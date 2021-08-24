import numpy as np
from flask import Flask, request, render_template
import joblib
import pickle
from textblob import TextBlob
# from myfunctions import *


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
        text = request.form['Review']
        y=text
        edu=TextBlob(y)
        x=edu.sentiment.polarity
        y=edu.sentiment.subjectivity

        data = [text]
        vectorizer = cv.transform(data).toarray()
        prediction = model.predict(vectorizer)
        print(prediction)
        
        if prediction == 0 :
            sentiment = 'Negative'
        elif prediction== 1 or (x==0 and y==0) :
            sentiment = 'Neutral'
        elif prediction==  2:
            sentiment='Positive'
        
        
        # message= "Reviews required to achieve Ideal score 4.5="
        # NumberOfReviews=Number_of_reviews_required()
        # message1= "Number of reviews for 0.5 drop="
        # NumberOfReviewsForDrop=Number_of_reviews_for_Drop()
    return render_template('index.html', text=text, prediction_text=sentiment)

if __name__ == "__main__":
    app.run(debug=True)