# IR13A.py CS5154/6054 cheng 2022
# use the first and the last 1000 lines of bible.txt as two classes
# find top terms according to mutual information
# Usage: python IR13A.py

import numpy as np
import sklearn.feature_selection as fs
from sklearn.feature_extraction.text import CountVectorizer

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N =len(docs)
trainX = np.concatenate([docs[0:1000], docs[N-1000:N]])
y = np.concatenate([np.zeros(1000, dtype=np.int16), np.ones(1000, dtype=np.int16)])

cv = CountVectorizer(binary=True, max_df=0.4, min_df=4)
X = cv.fit_transform(trainX).toarray()
print(X.shape)
voc = np.array(cv.get_feature_names())

mi = fs.chi2(X, y)
sorted = np.argsort(mi)[::-1]
# for i in range(10):
#     index = sorted[i]
#     print(voc[index], mi[index])

kbest = fs.SelectKBest(fs.chi2)
kbest.fit(X, y)
support = np.array(kbest.get_support())
print(voc[support])
