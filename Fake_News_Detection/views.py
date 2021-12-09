"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template,request
from Fake_News_Detection import app

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/detect')
def detect():
    """Renders the about page."""
    return render_template(
        'detect.html',
        title='Fake News Detection',
        year=datetime.now().year,
        message='Your application description page.'
    )


@app.route('/predictions',methods=['POST','GET'])
def predictions():
    if request.method == "POST":
        import numpy as np
        import pandas as pd
        import itertools
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import PassiveAggressiveClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix


        #Read the data
        df=pd.read_csv('news.csv')
        #Get shape and head
        df.shape
        df.head()

        #DataFlair - Get the labels
        labels=df.label
        labels.head()

        #Split the dataset
        x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


        #Initialize a TfidfVectorizer
        tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

        #Fit and transform train set, transform test set
        tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
        tfidf_test=tfidf_vectorizer.transform(x_test)



        #Initialize a PassiveAggressiveClassifier
        pac=PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train,y_train)

        #Predict on the test set and calculate accuracy
        y_pred=pac.predict(tfidf_test)
        score=accuracy_score(y_test,y_pred)
        print(f'Accuracy: {round(score*100,2)}%')


        #Build confusion matrix
        confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])



        #Detection
        def  ClassifyNews(newstext):
            newstext = pd.Series(newstext)
            vec_newtest=tfidf_vectorizer.transform(newstext)
            t_pred=pac.predict(vec_newtest)
            t_pred.shape
            return t_pred 
        
        news = request.form.get('news',type=str)

        predicted_result = ClassifyNews(news)
        
        return render_template('NewsResult.html', result = predicted_result[0],score = score*100)

