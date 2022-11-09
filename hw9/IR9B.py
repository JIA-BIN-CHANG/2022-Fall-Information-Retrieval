# IR9B.py CS5154/6054 cheng 2022
# CountVectorizer (binary=True) is used to make the term-document matrix A
# cosine similarity is used to find normalized co-occurrence of pairs of terms
# another cosine similarity is used on the first cosine similarity matrix
# Usage: python IR9B.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open("bible.txt", "r")
docs = f.readlines()
f.close()

cv = CountVectorizer(binary=True, max_df=0.04, min_df=8)
A = cv.fit_transform(docs).toarray().T
voc = cv.get_feature_names()
voclen, doclen = A.shape
r = cosine_similarity(A, A) # first order co-occurrence in r
np.fill_diagonal(r, 0)

s = cosine_similarity(r, r) # second order co-occurrence in s
np.fill_diagonal(s, 0)

for index in np.argsort(s.flatten())[::-1]:
    a = int(index / voclen)
    b = index % voclen
    if a > b:  # add condition there is no first-order co-occurrence between a and b
        if s[a][b] < 0.4:
            break
        if s[a][b] > 0.4 and r[a][b] < 0.4:
            print(s[a][b], voc[a], voc[b])