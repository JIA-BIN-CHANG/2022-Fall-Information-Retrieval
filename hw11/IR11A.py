# IR11A.py CS5154/6054 cheng 2022
# TfidfVectorizer is used to generate vocabulary
# a random term is the query and the top 5 cosine similarity
# in Tfidf are considered as the initial pseudo relevant for
# probabilistic pseudo relevance feedback (IIR 11.3.4)
# then pt, ut, ct are computed for terms and documents are ranked
# with sum of ct for t in docs 
# Usage: python IR11A.py

import re
import numpy as np
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pseudorelevant = 5

f = open("bible.txt", "r")
docs = f.readlines()
f.close()

tfidf = TfidfVectorizer(binary=True, max_df=0.04, min_df=8)
dt = tfidf.fit_transform(docs)
docsets = [set(d.indices) for d in dt]

N = len(docs)
terms = list(tfidf.vocabulary_)
T = len(terms)

query = random.choice(terms)
print(query, tfidf.vocabulary_.get(query))

q = tfidf.transform([query])
sim = cosine_similarity(q[0], dt)
relevantset = set()
for d in np.argsort(sim[0])[::-1][0:pseudorelevant]:
    print(sim[0][d], docs[d])
    relevantset.add(d)

print(relevantset)

for i in range(3):

    dfs = np.zeros(T, dtype=int)
    for d in range(N):
        for t in docsets[d]:
            dfs[t] = dfs[t] + 1

    ct = np.zeros(T)
    for t in range(T):
        Vt = 0
        for d in relevantset:
            if t in docsets[d]:
                Vt = Vt + 1
        pt = (Vt + 0.5)/(pseudorelevant + 1.0)
        ut = (dfs[t] - Vt + 0.5)/(N - pseudorelevant + 1.0)
        ct[t] = math.log(pt/(1 - pt) * (1 - ut)/ut)


    rsv = np.zeros(N)
    for d in range(N):
        for t in docsets[d]:
            rsv[d] = rsv[d] + ct[t]

    relevantset.clear()
    for d in np.argsort(rsv)[::-1][0:pseudorelevant]:
        print(rsv[d], docs[d])
        relevantset.add(d)

    print(relevantset)