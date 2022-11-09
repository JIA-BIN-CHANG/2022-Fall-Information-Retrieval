# IR14A.py CS5154/6054 cheng 2022
# Comparing MNB (multinomialNB) with binarized MNB and BernoulliNB
# on two random segments of bible.txt from the first third and last third
# 100 test documents are at the center of 1000 training documents
# Usage: python IR14A.py

import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
for i in range(10):
    N =len(docs)
    N1 = N // 3 - 1100
    c0 = random.randrange(N1)
    c1 = N - 1100 - random.randrange(N1)
    print(f'Current iteration: {i+1}, Random segments at -', c0, c1)

    trainX = np.concatenate([docs[c0:c0+500], docs[c0+600:c0+1100],
    docs[c1:c1+500], docs[c1+600:c1+1100]])
    y = np.concatenate([np.zeros(1000, dtype=np.int16), np.ones(1000, dtype=np.int16)])
    testX = np.concatenate([docs[c0+500:c0+600], docs[c1+500:c1+600]])
    testY = np.concatenate([np.zeros(100, dtype=np.int16), np.ones(100, dtype=np.int16)])

    cv = CountVectorizer(max_df=0.4, min_df=4)
    X = cv.fit_transform(trainX).toarray()
    T = cv.transform(testX).toarray()

    model = MultinomialNB()
    model.fit(X, y)
    pred = model.predict(T)
    print ('MNB Accuracy - ', accuracy_score(testY, pred))

    cv = CountVectorizer(binary=True, max_df=0.4, min_df=4)
    X = cv.fit_transform(trainX).toarray()
    T = cv.transform(testX).toarray()

    model = MultinomialNB()
    model.fit(X, y)
    pred = model.predict(T)
    print ('Binaruized MNB Accuracy - ', accuracy_score(testY, pred))

    model = BernoulliNB()
    model.fit(X, y)
    pred = model.predict(T)
    print ('BernoulliNB Accuracy - ', accuracy_score(testY, pred))

