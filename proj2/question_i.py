from common import *
from sklearn import metrics
from sklearn import linear_model
from util import *
import numpy as np

print "Logistic regression with regularization..."
print ""

t0 = time.time()

c = [0.001,0.1,10,1000,1e5]

for c_value in c:
    logreg1 = linear_model.LogisticRegression(C = c_value,penalty = 'l1')
    logreg1.fit(X_train_svd,train_group)
    logreg_predict1 = logreg1.predict(X_test_svd)

    print "l1 Logistic regression: c = {}, accuracy = {}, coeff[0] = {:0.5f}".format(c_value, np.mean(logreg_predict1 == test_group),logreg1.coef_[0][0])
    #print "coeff = %s\n" % str([float("{:0.5f}".format(x)) for x in logreg1.coef_[0]])

print ""

for c_value in c:
    logreg2 = linear_model.LogisticRegression(C = c_value,penalty = 'l2')
    logreg2.fit(X_train_svd,train_group)
    logreg_predict2 = logreg2.predict(X_test_svd)

    print "l2 Logistic regression: c = {}, accuracy = {}, coeff[0] = {:0.5f}".format(c_value, np.mean(logreg_predict2 == test_group),logreg2.coef_[0][0])
    #print "coeff = %s\n" % str([float("{:0.5f}".format(x)) for x in logreg2.coef_[0]])



print ""
util.print_time_in_miliseconds(t0, time.time())
