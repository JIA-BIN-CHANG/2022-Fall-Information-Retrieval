# IR4A.py CS5154/6054 cheng 2022
# read names from 2010 USA census surnames occurring 1000 times or more
# and have length 5 or more
# make the inverted index of 3grams
# perform spelling check by Jaccard coefficient
# Usage: python IR4A.py

f = open("names5.txt", "r")
names = [line.rstrip() for line in f]
f.close()
invertedIndex = {}  # building the inverted index for 3grams
for i in range(len(names)):
    name = names[i]
    grams3 = len(name) - 2
    for j in range(grams3):
        s = name[j:j+3]
        if invertedIndex.get(s) == None:
            invertedIndex.update({s : {i}})
        else:
            invertedIndex.get(s).add(i)
       
for i in range(10): # until user enters empty line
    query = input("Enter a name with at least five characters: ").lower()
    if len(query) < 3:
        break
    grams = set() # set of 3grams of the query
    qlen = len(query) - 2
    for j in range(qlen):
        grams.add(query[j:j+3])
    intersections = {}  # intersection counts, can be done with a Counter, too
    for t in grams: 
        if invertedIndex.get(t) == None:
            continue
        for d in invertedIndex.get(t):
            if intersections.get(d) == None:
                intersections.update({d : 1})
            else:
                x = intersections.get(d) + 1
                intersections.update({d : x})

    for k, v in intersections.items():
        jc = v /(grams3 + k - v)  # pseudocode, need sizes of the two sets
        if jc > 0.3:  # 0.3 is an arbitrary threshold
            print(jc, names[k])


