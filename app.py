from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk  
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from pickle import dump, load

app = Flask(__name__)
model = load(open("G:\Final yr Project\MNB.pickle", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_text')
def process_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i)) + " "
    return words

@app.route('/predict', methods=['POST'])
def predict():
    
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
    return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.debug = True
    app.run()