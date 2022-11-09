# IR15A.py CS5154/6054 cheng 2022
# Comparing classifiers on documents as binary, count, or tfidf vector
# on two random segments of bible.txt from the first third and last third
# 100 test documents are at the center of 1000 training documents
# Only the four in IIR Chapters 13 and 14 are implemented
# you need to add the ten others imported, too
# Usage: python IR15A.py

import numpy as np
import random
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
from sklearn.metrics import accuracy_score

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N =len(docs)
N1 = N // 3 - 1100
c0 = random.randrange(N1)
c1 = N - 1100 - random.randrange(N1)
print('Random segments at -', c0, c1)

trainX = np.concatenate([docs[c0:c0+500], docs[c0+600:c0+1100],
docs[c1:c1+500], docs[c1+600:c1+1100]])
y = np.concatenate([np.zeros(1000, dtype=np.int16), np.ones(1000, dtype=np.int16)])
testX = np.concatenate([docs[c0+500:c0+600], docs[c1+500:c1+600]])
testY = np.concatenate([np.zeros(100, dtype=np.int16), np.ones(100, dtype=np.int16)])

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
