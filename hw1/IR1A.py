# IR1A.py CS5154/6054 cheng 2022
# read lines from a text file
# retrieve documents for boolean queries
# Usage: python IR1A.py

import time

t1 = time.process_time_ns()
f = open("bible.txt", "r")
docs = f.readlines()
f.close()
t2 = time.process_time_ns()
       
word1 = 'punishment'
word2 = 'transgressions'
for i in range(100):
  for i in range(len(docs)):
      if word1 in docs[i] and word2 in docs[i]:
          print(i, docs[i])
t3 = time.process_time_ns()

print('Time analysis of IR1A.py')
print(f'Reading the collection of documents time: {t2-t1} ns')
print(f'Boolean retrieval with an expression time: {t3-t2} ns')