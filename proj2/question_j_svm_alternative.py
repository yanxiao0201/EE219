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

class OneVsOneClassifier:
	def __init__(self, train_dict, cat1, cat2):
		self.cat1 = cat1
		self.cat2 = cat2
		
		count_vect = text.CountVectorizer(min_df=1, stop_words='english', analyzer = 'word', tokenizer=util.my_tokenizer)
		tfidf_transformer = text.TfidfTransformer()
		svd = TruncatedSVD(n_components=50, algorithm='arpack')

		X_train_counts = count_vect.fit_transform(train_dict[cat1].data + train_dict[cat2].data)
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
		X_train_svd = svd.fit_transform(X_train_tfidf)

		y_train = [1] * len(train_dict[cat1].filenames) + [-1] * len(train_dict[cat2].filenames)

		self.count_vect = count_vect
		self.tfidf_transformer = tfidf_transformer
		self.svd = svd

		self.clf = svm.LinearSVC(random_state=42, class_weight='balanced')
		self.clf.fit(X_train_svd, y_train)

	def predict(self, mat):
		X_test_counts = self.count_vect.transform(mat)
		X_test_tfidf = self.tfidf_transformer.transform(X_test_counts)
		X_test_svd = self.svd.transform(X_test_tfidf)

		raw_pred = self.clf.predict(X_test_svd)
		named_pred = []
		for idx, pred in enumerate(raw_pred):
			cat = self.cat1 if pred >= 0 else self.cat2
			named_pred.append(cat)
		return named_pred

#--------------------------------

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

print "Loading data..."
t0 = time.time()

newsgroup_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

newsgroup_train_dict = {}
for cat in categories:
	newsgroup_train_dict[cat] = fetch_20newsgroups(subset='test', categories = [cat], shuffle=True, random_state=42)

util.print_time_in_miliseconds(t0, time.time())

##############
# ONE VS ONE #
##############

print "[One VS One]"

classifiers_1v1 = []
for idx, cat1 in enumerate(categories):
	for cat2 in categories[idx+1:]:
		print "Building 1v1 classifier: %s vs %s" % (cat1, cat2)
		t0 = time.time()
		clf = OneVsOneClassifier(newsgroup_train_dict, cat1, cat2)
		classifiers_1v1.append(clf)
		util.print_time_in_miliseconds(t0, time.time())

preds = []
for clf in classifiers_1v1:
	print "Classifier (%s vs %s) is predicting..." % (clf.cat1, clf.cat2)
	t0 = time.time()
	preds.append(clf.predict(newsgroup_test.data))
	util.print_time_in_miliseconds(t0, time.time())
preds = np.array(preds)

y_pred = []
y_true = [fname.split('/')[-2] for fname in newsgroup_test.filenames]
for idx, row in enumerate(preds.T):
	ballot = []
	for cat in categories:
		ballot.append(len([1 for pred in row if pred == cat]))
	index, maxval = max(enumerate(ballot), key=operator.itemgetter(1))
	#if len([1 for val in ballot if val == maxval]) > 1:
	#	print "Warning: there are 2 or more choices in row %d: %s" % (idx, row)
	y_pred.append(categories[index])

y_pred = [categories.index(val) for val in y_pred]
y_true = [categories.index(val) for val in y_true]
print "Accuracy (overall): %.6f" % util.get_accuracy(y_pred, y_true)
print metrics.classification_report(y_true, y_pred, target_names=categories)
print "Confusion Matrix:"
print metrics.confusion_matrix(newsgroup_test.target, y_pred)

#--------------------------------

###################
# ONE VS THE REST #
###################

