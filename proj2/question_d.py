from question_b import *
from sklearn.decomposition import TruncatedSVD

print "SVD..."
t0 = time.time()

svd = TruncatedSVD(n_components=50, algorithm='arpack')

X_train_svd = svd.fit_transform(X_train_tfidf)
print X_train_svd.shape
#print X_train_svd[0:100]
util.print_time_in_miliseconds(t0, time.time())

