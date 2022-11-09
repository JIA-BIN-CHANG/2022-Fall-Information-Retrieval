# IR7A.py CS5154/6054 cheng 2022
# read lines from a text file as docs
# tokenize each as a bag of words
# make the inverted index
# randomly select a doc as the query with at least 5 words
# using the inverted index
# retrieve sets of docs containing each of the word in query
# update a dictionary called intersection with tf-idf
# call the top 10 tf-idf documents "relevant"
# then call documents with the top 10 Jaccard coefficient "retrieved"
# compute precision, recall, and F1
# they are all the same, why?
# Usage: python IR7A.py

import re
import numpy as np
import random
import math
from collections import Counter
from heapq import nlargest

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N = len(docs)
logN = math.log(N)

docLen = np.zeros(N, dtype=int)
counter = Counter()
invertedIndex = {}
for i in range(N):
    tokens = re.findall('\w+', docs[i])
    docLen[i] = len(tokens)
    counter.clear()
    counter.update(tokens)
    for t, tf in counter.items():
        if invertedIndex.get(t) == None:
            invertedIndex.update({t : {i : counter.get(t)}})
        else:
            invertedIndex.get(t).update({i : counter.get(t)})
       
p_list=[]
r_list = []
f1_list = []
for i in range(5):
    query = random.randint(0, N)
    print(query, docs[query])
    tokens = re.findall('\w+', docs[query])
    qlen = len(tokens)
    counter.clear()
    counter.update(tokens)
    print(counter)

    intersections = {}
    jaccard = {}
    for t, tf1 in counter.items():  
        idf = logN - math.log(len(invertedIndex.get(t)))
        for d, tf2 in invertedIndex.get(t).items():
            tfidf = tf1 * idf * tf2 * idf
            if intersections.get(d) == None:
                intersections.update({d : tfidf})
                jaccard.update({d : 1})
            else:
                x = intersections.get(d) + tfidf
                intersections.update({d : x})
                y = jaccard.get(d) + 1
                jaccard.update({d : y})

    relevant = set(nlargest(10, intersections, key = intersections.get))
    print('relevant', relevant)

    for d, ab in jaccard.items():
        jaccard.update({d : ab / (qlen + docLen[d] - ab)})
    retrieved = set(nlargest(10, jaccard, key = jaccard.get))
    print('retrieved', retrieved)
    # your code for precision, recall, and f1
    tp=[]
    fp=[]
    tn=[]
    fn=[]
    for item in retrieved:
        if item in relevant:
            tp.append(item)
    for item in retrieved:
        if item not in relevant:
            fp.append(item)
    for item in intersections.items():
        if item[0] not in retrieved and item[0] in relevant:
            fn.append(item)
    for item in intersections.items():
        if item[0] not in retrieved and item[0] not in relevant:
                tn.append(item[0])
    p = len(tp)/(len(tp)+len(fp))
    r = len(tp)/(len(tp)+len(fn))
    f1 = 2/((1/p)+(1/r))
    p_list.append(p)
    r_list.append(r)
    f1_list.append(f1)

print(f'Precision list: {p_list}')
print(f'Recall list: {r_list}')
print(f'F1 Score list: {f1_list}')
print(f'Average Precision is: {sum(p_list)/len(p_list)}')
print(f'Average Recall is: {sum(r_list)/len(r_list)}')
print(f'Average F1-Score is: {sum(f1_list)/len(f1_list)}')
