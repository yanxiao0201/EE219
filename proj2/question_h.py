from common import *
from sklearn import metrics
from sklearn import linear_model
from util import *

print "Logistic regression..."
print ""

t0 = time.time()

logreg = linear_model.LogisticRegression(C = 1e5)
logreg.fit(X_train_svd,train_group)
logreg_predict = logreg.predict(X_test_svd)

# calculate the class probabilitie
logreg_score = logreg.predict_proba(X_test_svd)
target_names = ['comp', 'rec']

pos_score = np.array([score[1] for score in logreg_score])

metric_analysis_decfun(test_group, logreg_predict, target_names, pos_score, 'logistic_regression')

print ""
util.print_time_in_miliseconds(t0, time.time())
