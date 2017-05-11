from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import svm
from sklearn import metrics
import numpy as np
import time
import operator
import util

#--------------------------------

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

print "Loading data..."
t0 = time.time()

newsgroup_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
newsgroup_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

util.print_time_in_miliseconds(t0, time.time())

#--------------------------------

print "Vectorizing..."
t0 = time.time()

count_vect = text.CountVectorizer(min_df=1, stop_words='english', analyzer = 'word', tokenizer=util.my_tokenizer)
tfidf_transformer = text.TfidfTransformer()
svd = TruncatedSVD(n_components=50, algorithm='arpack')

X_train_counts = count_vect.fit_transform(newsgroup_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_svd = svd.fit_transform(X_train_tfidf)

X_test_counts = count_vect.transform(newsgroup_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_svd = svd.transform(X_test_tfidf)

util.print_time_in_miliseconds(t0, time.time())

print ""

#--------------------------------

##############
# ONE VS ONE #
##############

print "[One VS One]"
t0 = time.time()

clf_1v1 = OneVsOneClassifier(svm.LinearSVC(random_state=42, class_weight='balanced'))
clf_1v1.fit(X_train_svd, newsgroup_train.target)
y_pred = clf_1v1.predict(X_test_svd)

util.print_time_in_miliseconds(t0, time.time())

y_true = newsgroup_test.target
print "Accuracy (overall): %.6f" % util.get_accuracy(y_pred, y_true)
print metrics.classification_report(y_true, y_pred, target_names=categories)
print "Confusion Matrix:"
print metrics.confusion_matrix(newsgroup_test.target, y_pred)

print ""

#--------------------------------

###################
# ONE VS THE REST #
###################

print "[One Vs Rest]"

clf_1vr = OneVsRestClassifier(svm.LinearSVC(random_state=42))
clf_1vr.fit(X_train_svd, newsgroup_train.target)
y_pred = clf_1vr.predict(X_test_svd)

util.print_time_in_miliseconds(t0, time.time())

print "Accuracy (overall): %.6f" % util.get_accuracy(y_pred, y_true)
print metrics.classification_report(y_true, y_pred, target_names=categories)
print "Confusion Matrix:"
print metrics.confusion_matrix(newsgroup_test.target, y_pred)
