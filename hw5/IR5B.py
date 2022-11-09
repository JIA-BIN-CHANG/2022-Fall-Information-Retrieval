# IR5B.py CS5154/6054 cheng 2022
# read lines from a text file as docs
# tokenize each as a bag of words
# make the inverted index
# randomly select a doc as the query with at least 5 words
# using the inverted index
# retrieve sets of docs containing each of the word in query
# update a dictionary called intersection with tf-idf
# display the top 5 documents
# Usage: python IR5B.py

import re
import random
import math
from collections import Counter
from heapq import nlargest

f = open("bible.txt", "r")
docs = f.readlines()
f.close()

counter = Counter()
invertedIndex = {}
for i in range(len(docs)):
    counter.clear()
    counter.update(re.findall('\w+', docs[i]))
    for t, tf in counter.items():
        if invertedIndex.get(t) == None:
            invertedIndex.update({t : {i : counter.get(t)}})
        else:
            invertedIndex.get(t).update({i : counter.get(t)})
       
N = len(docs)
logN = math.log(N)
query = random.randint(0, N)
print(query, docs[query])
counter.clear()
counter.update(re.findall('\w+', docs[query]))
print(counter)

intersections = {}
for t, tf1 in counter.items():  
    idf = logN - math.log(len(invertedIndex.get(t)))
    print(t, idf)

    for d, tf2 in invertedIndex.get(t).items():
        tfidf = (1 + math.log(tf1)) * idf * (1 + math.log(tf2)) * idf
        if intersections.get(d) == None:
            intersections.update({d : tfidf})
        else:
            x = intersections.get(d) + tfidf
            intersections.update({d : x})

print()
res = nlargest(5, intersections, key = intersections.get)
for k in res:
    print(k, intersections.get(k), docs[k])
