# IR16B.py CS5154/6054 cheng 2022
# three classes
# Usage: python IR16B.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N =len(docs)
mid = 1100 + (N - 3 * 1100) // 2
trainX = np.concatenate([docs[0:1000], docs[mid:mid+1000], docs[N-1000:N]])
y = np.concatenate([np.zeros(1000, dtype=np.int16), np.ones(1000, dtype=np.int16), np.full(1000, 2, dtype=np.int16)])
testX = np.concatenate([docs[1000:1100], docs[mid+1000:mid+1100], docs[N-1100:N-1000]])
testY = np.concatenate([np.zeros(100, dtype=np.int16), np.ones(100, dtype=np.int16), np.full(100, 2, dtype=np.int16)])

model_list = [KNeighborsClassifier(), NearestCentroid(), LogisticRegression(), LinearSVC(), SVC(), DecisionTreeClassifier(), ExtraTreeClassifier(), 
                ExtraTreesClassifier(), AdaBoostClassifier(), RandomForestClassifier(), Perceptron(), MLPClassifier()]
model_name = ['KNN', 'Rocchio', 'LogisticRegression', 'LinearSVC', 'SVC', 'DecisionTreeClassifier', 'ExtraTreeClassifier', 
                'ExtraTreesClassifier', 'AdaBoostClassifier', 'RandomForestClassifier', 'Perceptron', 'MLPClassifier']

for i in range(5):
    # documents as binary vectors
    cv = CountVectorizer(binary=True, max_df=0.4, min_df=4)
    X0 = cv.fit_transform(trainX).toarray()
    T0 = cv.transform(testX).toarray()

    # documents as count vectors
    cv = CountVectorizer(max_df=0.4, min_df=4)
    X1 = cv.fit_transform(trainX).toarray()
    T1 = cv.transform(testX).toarray()

    # documents as tfidf vectors
    cv = TfidfVectorizer(max_df=0.4, min_df=4)
    X2 = cv.fit_transform(trainX).toarray()
    T2 = cv.transform(testX).toarray()

    highest = 0
    highest_result = []

    model = BernoulliNB()
    model.fit(X0, y)
    A0 = accuracy_score(testY, model.predict(T0))
    print ('BernoulliNB -', A0)
    if A0 >= highest:
        highest_result = ['BernoulliNB', A0, 'binary vector']
        highest = A0

    model = MultinomialNB()
    model.fit(X0, y)
    A0 = accuracy_score(testY, model.predict(T0))
    model.fit(X1, y)
    A1 = accuracy_score(testY, model.predict(T1))
    print ('MultinomialNB -', A0, A1)
    candidates = [(A0, 'binary vector'), (A1, 'count vector')]
    for candidate in candidates:
        if candidate[0] >= highest:
            highest_result = ['MultinomialNB', candidate[0], candidate[1]]
            highest = candidate[0]

    for index, model in enumerate(model_list):
        model.fit(X0, y)
        A0 = accuracy_score(testY, model.predict(T0))
        model.fit(X1, y)
        A1 = accuracy_score(testY, model.predict(T1))
        model.fit(X2, y)
        A2 = accuracy_score(testY, model.predict(T2))
        print (model_name[index] + ' -', A0, A1, A2)
        candidates = [(A0, 'binary vector'), (A1, 'count vector'), (A2, 'tfidf vector')]
        for candidate in candidates:
            if candidate[0] >= highest:
                highest_result = [model_name[index], candidate[0], candidate[1]]
                highest = candidate[0]
    print(f'Model is : {highest_result[0]}, Highest accuracy is : {highest_result[1]}, vectorizer is : {highest_result[2]}')
