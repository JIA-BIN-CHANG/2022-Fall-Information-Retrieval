# IR8A.py CS5154/6054 cheng 2022
# TfidfVectorizer and CountVectorizer (binary=True) are used 
# a random doc is the query and the top 50 cosine similarity
# in Tfidf are considered relevent
# CountVectors are ranked using cosine similarity
# precision and recall at each retrieval level are computed
# and the precision-recall graph (Fig 8.2 iir) is plotted
# Usage: python IR8A.py

import re
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
relevant = 50

f = open("bible.txt", "r")
docs = f.readlines()
f.close()

tfidf = TfidfVectorizer(max_df=0.4, min_df=2)
dt = tfidf.fit_transform(docs)
N = len(docs)
query = random.randint(0, N)
print(query, docs[query])

sim = cosine_similarity(dt[query], dt)
toptfidf = set()
for index in np.argsort(sim)[0][::-1][0:relevant]:
    toptfidf.add(index)

print(toptfidf)

cv = CountVectorizer(binary=True, max_df=0.4, min_df=2)
dt2 = cv.fit_transform(docs)
sim2 = cosine_similarity(dt2[query], dt2)
sorted = np.argsort(sim2)[0][::-1]
precision = np.zeros(N)
recall = np.zeros(N)
m = 0
for i in range(N):
    if sorted[i] in toptfidf:
        m = m + 1
    tp = m
    fn = relevant - m
    fp = i + 1 - m
    tn = N - tp - fn - fp
    precision[i] = tp / (tp + fp)
    recall[i] = tp / (tp + fn)

# correct AP calculation
# first append sentinel values at the end
mrec = np.concatenate(([0.], recall, [1.]))
mpre = np.concatenate(([0.], precision, [0.])) 

# compute the precision envelope
for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

# to calculate area under PR curve, look for points
# where X axis (recall) changes value
i = np.where(mrec[1:] != mrec[:-1])[0]

# AP= AP1 + AP2+ AP3+ AP4 + ...
ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

print(f'MAP: {ap}')

plt.figure()
plt.title("precision-recall graph")
plt.scatter(recall, precision)

eleven_recalls = np.zeros(11)
interpolated = np.zeros(11)
n =0
for i in range(N):
    if n <= 10 and recall[i] * 10 >= n: 
        interpolated[n] = max(precision[i:]) 
        eleven_recalls[n] = recall[i]
        n =n + 1
    if n > 10: break
plt.figure()
plt.title("eleven-point interpolated precision-recall graph")
plt.scatter(eleven_recalls, interpolated)

rocx = np.zeros(N)
recall = np.zeros(N)
m= 0
for i in range(N):
    if sorted[i] in toptfidf:
        m= m+ 1
        tp = m
        fn = relevant - m
        fp = i + 1 - m
        tn = N - tp - fn - fp
        rocx[i] = fp / (fp + tn)
        recall[i] = tp / (tp + fn)
plt.figure()
plt.title("ROC curve")
plt.scatter(rocx, recall)

plt.show()