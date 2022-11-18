# IR21A.py CS5154/6054 cheng 2022
# decompose the document-term matrix
# Usage: python IR21A.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from matplotlib import pyplot as plt

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N = len(docs)

vectorizer = TfidfVectorizer(max_df=1000, min_df=100)
X = vectorizer.fit_transform(docs)
words = vectorizer.get_feature_names()

# nmf_model = NMF(n_components=5, init='nndsvda')
# W = nmf_model.fit_transform(X).T
# H = nmf_model.components_

svd_model = TruncatedSVD(n_components=5)
W = svd_model.fit_transform(X).T
H = svd_model.components_

# lda_model = LatentDirichletAllocation(n_components=5)
# W = lda_model.fit_transform(X).T
# H = lda_model.components_

from wordcloud import WordCloud

for topic in range(5):
    size = {}
    largest = H[topic].argsort()[::-1] 
    for i in range(40):
        size[words[largest[i]]] = abs(H[topic][largest[i]])
    wc = WordCloud(background_color="white", max_words=100, width=960, height=540)
    wc.generate_from_frequencies(size)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

