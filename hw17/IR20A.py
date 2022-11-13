# IR20A.py CS5154/6054 cheng 2022
# HAC with four different linkage modes
# display confusion matrix and NMI between the clusterings
# Usage: python IR20A.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, normalized_mutual_info_score
import matplotlib.pyplot as plt
import itertools as it

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

average = AgglomerativeClustering(n_clusters=4, linkage='average')
average.fit_predict(X)

ward = AgglomerativeClustering(n_clusters=4, linkage='ward')
ward.fit_predict(X)

sets = [single, complete, ward, average]
names = ['single', 'complete', 'ward', 'average']
sets_combinations = it.combinations(sets, 2)
sets_combinations = list(sets_combinations)
names_combinations = it.combinations(names, 2)
names_combinations = list(names_combinations)

for set, name in zip(sets_combinations, names_combinations):
    cm = confusion_matrix(set[0].labels_, set[1].labels_)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    nmi_score = normalized_mutual_info_score(set[0].labels_, set[1].labels_)
    print(f'NMI score between {name[0]} and {name[1]} is: {nmi_score}')
plt.show()