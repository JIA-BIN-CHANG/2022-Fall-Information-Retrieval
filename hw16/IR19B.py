# IR19B.py CS5154/6054 cheng 2022
# twice k-means
# confusion matrix
# NMI, RI, and purity
# Usage: python IR19B.py

import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, normalized_mutual_info_score, rand_score
import matplotlib.pyplot as plt

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N = 1000
firstk = docs[0:N]

cv = TfidfVectorizer(max_df=0.4, min_df=3)
X = cv.fit_transform(firstk)

model = KMeans(n_init=1, max_iter=10)
model.fit_predict(X)
y1 = model.labels_

model.fit_predict(X)
y2 = model.labels_

cm = confusion_matrix(y1, y2)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

cm2 = cm * 2 # Exercise 16.3: replace every point d with two identical copies

# You code to compute and print NMI, RI, and purity(in two ways)
# see slide 11/3/15 for purity calculation
# Then run the same on cm2 so you get the answer to Exercise 16.3 (ii)
