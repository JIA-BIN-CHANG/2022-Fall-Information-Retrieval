# IR14C.py CS5154/6054 cheng 2022
# NB top features by parameter weights
# Usage: python IR14C.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N =len(docs)
trainX = np.concatenate([docs[0:1000], docs[N-1000:N]])
y = np.concatenate([np.zeros(1000, dtype=np.int16), np.ones(1000, dtype=np.int16)])

cv = CountVectorizer(binary=True, max_df=0.4, min_df=4)
X = cv.fit_transform(trainX).toarray()
print(X.shape)
voc = cv.get_feature_names()

model = MultinomialNB()
model.fit(X, y)
logprob = model.feature_log_prob_
logodds = logprob[0] - logprob[1]
vocpos = logodds.argsort()
for i in vocpos[:10]:
    print(voc[i], logodds[i])
for i in vocpos[-10:]:
    print(voc[i], logodds[i])
