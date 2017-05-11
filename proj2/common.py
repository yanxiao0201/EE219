from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import time
import util

print "Loading comp and rec data..."
t0 = time.time()
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
newsgroup_train = fetch_20newsgroups(subset='train', categories = categories, shuffle=True, random_state=42)
newsgroup_test = fetch_20newsgroups(subset='test', categories = categories, shuffle=True, random_state=42)

util.print_time_in_miliseconds(t0, time.time())

#--------------------------------
train_group = [0] * len(newsgroup_train.filenames)
test_group = [0] * len(newsgroup_test.filenames)
for group, newsgroup in [(train_group, newsgroup_train), (test_group, newsgroup_test)]:
	for idx, data in enumerate(newsgroup.filenames):
		fname = newsgroup.filenames[idx]
		class_name = fname.split('/')[-2]
		if 'comp.' in class_name:
			group[idx] = -1
		elif 'rec.' in class_name:
			group[idx] = 1

if 0 in train_group:
	raise Exception("Uncategorized data in training group!")
if 0 in test_group:
	raise Exception("Uncategorized data in testing group!")

#train_group = [0 if 'comp' in fname else 1 for fname in newsgroup_train.filenames]
#test_group = [0 if 'comp' in fname else 1 for fname in newsgroup_test.filenames]

#Vectorizer, TFxIDF and LSI
count_vect = text.CountVectorizer(min_df=1, stop_words='english', analyzer = 'word', tokenizer=util.my_tokenizer)
tfidf_transformer = text.TfidfTransformer()
svd = TruncatedSVD(n_components=50, algorithm='arpack')

print "Vectorizing and converting to TFxIDF... (training)"

X_train_counts = count_vect.fit_transform(newsgroup_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_svd = svd.fit_transform(X_train_tfidf)
print X_train_svd.shape


print "Vectorizing and converting to TFxIDF... (testing)"

X_test_counts = count_vect.transform(newsgroup_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_svd = svd.transform(X_test_tfidf)
print X_test_svd.shape
