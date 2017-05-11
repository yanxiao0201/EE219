from sklearn.feature_extraction import text
from sklearn import svm
from common import *
from util import *
from sklearn.model_selection import cross_val_score
import numpy as np

c = [0.001,0.01,0.1,1,10,100,1000]

print "soft margin SVM..."
print ""

t0 = time.time()

for c_value in c:
    clf = svm.LinearSVC(C = c_value)
    clf.fit(X_train_svd,train_group)

    sm_scores = cross_val_score(clf, X_test_svd, test_group, cv=5)
    ave_score = np.mean(sm_scores)
    print "C value = {}, ave_score = {}".format(c_value,ave_score)

print ""
#based on the result use C = 0.8 as the best value below:
clf = svm.LinearSVC(C = 0.8)
clf.fit(X_train_svd,train_group)
sm_SVM_predict = clf.predict(X_test_svd)

target_names = ['comp', 'rec']

print "Accuracy: {}".format(np.mean(sm_SVM_predict == test_group))
print ""
print "Classification report:"
print metrics.classification_report(test_group, sm_SVM_predict, target_names=target_names)
print ""
print "Confusion matrix:"
print metrics.confusion_matrix(test_group, sm_SVM_predict)
print ""
util.print_time_in_miliseconds(t0, time.time())
