
from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import (confusion_matrix, accuracy_score)
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    data = df.rename(columns={"v1":"label", "v2":"text"})

    from sklearn.preprocessing import label_binarize
    encoded_column_vector = label_binarize(data['label'], classes=['ham','spam']) # ham will be 0 and spam will be 1
    encoded_labels = np.ravel(encoded_column_vector) # Reshape array
    data['indicator'] = data.label.map({'spam':1,'ham':0 })


    from sklearn.model_selection import train_test_split
    # Continue as normal
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(data["text"], encoded_labels,
                                                      test_size = 0.1, random_state = 101)

    # Text Transformation
    from sklearn.feature_extraction.text import CountVectorizer

    vect = CountVectorizer()
    vect.fit(X_train_raw)

    X_train_df = vect.transform(X_train_raw)
    X_test_df = vect.transform(X_test_raw)

    from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict,
                                     StratifiedKFold, learning_curve)

    clf = LogisticRegression()

    clf.fit(X_train_df,y_train)

    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB


    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = vect.transform(data).toarray()
    	my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
