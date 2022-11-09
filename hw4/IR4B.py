# IR4B.py CS5154/6054 cheng 2022
# read names from 2010 USA census surnames occurring 1000 times or more
# and have length 8 or more
# make 3gram sets for all
# Usage: python IR4B.py

def shingles(name):  
    grams = set()
    for j in range(len(name) - 2):
        grams.add(name[j:j+3])
    return grams

f = open("names8.txt", "r")
names = [line.rstrip() for line in f]  # getting rid of the newline character
f.close()

N = len(names)
for i in range(N):  # looping all pairs, too slow on, say, tweets_01-08-2021
    A = shingles(names[i])
    Alen = len(A)
    for j in range(i + 1, N):
        B = shingles(names[j])
        intersection = A & B
        c = len(intersection)
        if c == 0:
            continue
        jc = c / (Alen + len(B) - c) # pseudocode
        if jc > 0.88:  # arbitrary threshold
            print(names[i], names[j], jc)

