from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import util

print "Loading comp and rec data..."
util.tic()

categories = ['comp.graphics',
			  'comp.os.ms-windows.misc',
			  'comp.sys.ibm.pc.hardware',
			  'comp.sys.mac.hardware',
			  'rec.autos',
			  'rec.motorcycles',
			  'rec.sport.baseball',
			  'rec.sport.hockey']
newsgroup = fetch_20newsgroups(subset='all', categories = categories, shuffle=True, random_state=42)

util.toc()

print "Vectorizing and converting to TFxIDF..."
util.tic()

count_vect = text.CountVectorizer(min_df=1, stop_words='english', analyzer = 'word', tokenizer=util.my_tokenizer)
tfidf_transformer = text.TfidfTransformer()

X_counts = count_vect.fit_transform(newsgroup.data)
X_tfidf = tfidf_transformer.fit_transform(X_counts)

util.toc()
