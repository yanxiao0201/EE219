from sklearn.feature_extraction import text
from sklearn import svm
from common import *
from util import *


print "SVM two groups..."
print ""

t0 = time.time()

clf = svm.LinearSVC(C=1e5)
clf.fit(X_train_svd,train_group)
svm_predict = clf.predict(X_test_svd)

svm_score = clf.decision_function(X_test_svd)

target_names = ['comp', 'rec']

metric_analysis_decfun(test_group, svm_predict, target_names, svm_score, 'SVM two groups')

print ""
util.print_time_in_miliseconds(t0, time.time())
