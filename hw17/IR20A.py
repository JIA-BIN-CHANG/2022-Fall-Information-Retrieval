# IR20A.py CS5154/6054 cheng 2022
# HAC with four different linkage modes
# display confusion matrix and NMI between the clusterings
# Usage: python IR20A.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, 
normalized_mutual_info_score
import matplotlib.pyplot as plt

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N = 1000
firstk = docs[0:N]

cv = TfidfVectorizer(max_df=0.4, min_df=3)
X = cv.fit_transform(firstk).toarray()

single = AgglomerativeClustering(n_clusters=4, linkage='single')
single.fit_predict(X)

complete = AgglomerativeClustering(n_clusters=4, linkage='complete')
complete.fit_predict(X)

cm = confusion_matrix(single.labels_, complete.labels_)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()