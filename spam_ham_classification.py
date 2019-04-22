import pandas as pd
import numpy as np
from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
data = pd.read_table("/root/Desktop/SMSSpamCollection.txt")
data = data.rename(index=str, columns={"ham": "ham/spam", "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...": "Message"})
train = data[:4400] # 4400 items
test = data[4400:] # 1172 items



y = data['ham/spam'].as_matrix()
X_text = data['Message'].as_matrix()

from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm


f = feature_extraction.text.CountVectorizer(stop_words = 'english')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(f,lowercase=True)
X = vectorizer.fit_transform(X_text).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.202, random_state=42)


##logistic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf = LogisticRegression()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
logistic_scor = accuracy_score(y_test,pred)


#naive bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
naive_bayes_scor= accuracy_score(y_test,pred)

#decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
decision_tree_scor = accuracy_score(y_test,pred)


#svm
#from sklearn.svm import SVC
#clf = SVC(gamma=0.1,C=1,kernel='rbf')
#clf.fit(X_train,y_train)
#pred = clf.predict(X_test)
#svm_scor = accuracy_score(y_test,pred)

##random_forest
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()
#clf.fit(X_train,y_train)
#pred = clf.predict(X_test)
#random_forest_scor = accuracy_score(y_test,pred)







import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas as pd

app = Flask(__name__)
global Classifier
global Vectorizer


# load data
data = pd.read_table("/root/Desktop/SMSSpamCollection.txt")
data = data.rename(index=str, columns={"ham": "v1", "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...": "v2"})
train_data = data[:4400] # 4400 items
test_data = data[4400:] # 1172 items



# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)







@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # show html form
        return '''
            <form method="post">
                <input type="text" name="message" />
                <input type="submit" value="Submit" />
            </form>
        '''
    
    
    
    elif request.method == 'POST':
        message = request.form.get('message')
        error = ''
        predict = ''
        logistic_score = ''
        naive_bayes_score = ''
        decision_tree_score = ''
        #svm_score = ''
        #random_forest_score = ''
        #adaboost_score = ''


        global Classifier
        global Vectorizer
        try:
            if len(message) > 0:
                vectorize_message = Vectorizer.transform([message])
                predict = Classifier.predict(vectorize_message)[0]
        except BaseException as inst:
            error = str(type(inst).__name__) + ' ' + str(inst)
        return jsonify(
                  message=message,
                  predict=predict, error=error,
                  logistic_score=logistic_scor,
                  naive_bayes_score=naive_bayes_scor,
                  decision_tree_score=decision_tree_scor)
                  #svm_score=svm_scor)
                  #random_forest_score=random_forest_scor
                  #adaboost_score=adaboost_scor)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5006))
app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
 
