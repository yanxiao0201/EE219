from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF
from sklearn import metrics
import numpy as np
import time
import util

print "Loading 4 groups of data..."
t0 = time.time()
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
newsgroup_train = fetch_20newsgroups(subset='train', categories = categories, shuffle=True, random_state=42)
newsgroup_test = fetch_20newsgroups(subset='test', categories = categories, shuffle=True, random_state=42)

util.print_time_in_miliseconds(t0, time.time())
#--------------------------------

#Vectorizer, TFxIDF and LSI
count_vect = text.CountVectorizer(min_df=1, stop_words='english', tokenizer=util.my_tokenizer)
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

#Naive Bayes classifier
#Use NMF because Naive Bayes only takes non-negative values
nmf = NMF(n_components=50)#, init='nndsvd')
X_train_nmf = nmf.fit_transform(X_train_tfidf)
X_test_nmf = nmf.transform(X_test_tfidf)

clf = MultinomialNB(alpha=0)
clf.fit(X_train_nmf, newsgroup_train.target) # train
predicted = clf.predict(X_test_nmf)
print np.mean(predicted == newsgroup_test.target)

print(metrics.classification_report(newsgroup_test.target, predicted, target_names=newsgroup_test.target_names))

print metrics.confusion_matrix(newsgroup_test.target, predicted)

util.print_time_in_miliseconds(t0, time.time())

