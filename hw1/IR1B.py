# IR1B.py CS5154/6054 cheng 2022
# read lines from a text file
# turn each line into a list of tokens
# make the inverted index
# retrieve documents using boolean queries
# Usage: python IR1B.py

import time

t1 = time.process_time_ns()
f = open("bible.txt", "r")
docs = f.readlines()
f.close()
t2 = time.process_time_ns()

invertedIndex = {}
for i in range(len(docs)):
    for s in docs[i].split():
        if invertedIndex.get(s) == None:
            invertedIndex.update({s : {i}})
        else:
            invertedIndex.get(s).add(i)
t3 = time.process_time_ns()
       
word1 = 'punishment'
word2 = 'transgressions'
for i in range(100):
  for j in invertedIndex.get(word1) & invertedIndex.get(word2):
      print(j, docs[j])
t4 = time.process_time_ns()

print('Time analysis of IR1B.py')
print(f'Reading the collection of documents time: {t2-t1} ns')
print(f'Making the inverted index time: {t3-t2} ns')
print(f'Boolean retrieval with an expression time: {t4-t3} ns')