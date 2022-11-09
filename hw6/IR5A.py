# IR5A.py CS5154/6054 cheng 2022
# read lines from a text file as docs
# tokenize each as a set of words
# make the inverted index
# randomly select a doc as the query with at least 5 words
# using the inverted index
# retrieve sets of docs containing each of the word in query
# update a dictionary called intersection with idf
# display the top 5 documents
# Usage: python IR5A.py

import re
import random
import math
from heapq import nlargest

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
invertedIndex = {}
for i in range(len(docs)):
    for s in set(re.findall('\w+', docs[i])):
        if invertedIndex.get(s) == None:
            invertedIndex.update({s : {i}})
        else:
            invertedIndex.get(s).add(i)
       
N = len(docs)
logN = math.log(N)
query = random.randint(0, N)
A = set(re.findall('\w+', docs[query]))
print(query, docs[query])

intersections = {}
for t in A:  
    idf = logN - math.log(len(invertedIndex.get(t)))
    print(t, idf)
    for d in invertedIndex.get(t):
        if intersections.get(d) == None:
            intersections.update({d : idf})
        else:
            x = intersections.get(d) + idf
            intersections.update({d : x})

print()
res = nlargest(5, intersections, key = intersections.get)
for k in res:
    print(k, intersections.get(k), docs[k])
