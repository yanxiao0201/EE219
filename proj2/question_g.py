from common import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF

print "Naive Bayes..."
t0 = time.time()

#Naive Bayes classifier
#Use NMF because Naive Bayes only takes non-negative values
nmf = NMF(n_components=50)
X_train_nmf = nmf.fit_transform(X_train_tfidf)
X_test_nmf = nmf.transform(X_test_tfidf)

clf = MultinomialNB()
clf.fit(X_train_nmf, train_group) # train
predicted = clf.predict(X_test_nmf)
scores = clf.predict_proba(X_test_nmf)
#print scores.shape
#print len(test_group)
#target_names = ['comp', 'rec']
#
#util.metric_analysis(test_group, predicted, target_names, scores, 'MultinomialNB')
# calculate the class probabilitie
target_names = ['comp', 'rec']

pos_score = np.array([score[1] for score in scores])

util.metric_analysis_decfun(test_group, predicted, target_names, pos_score, 'MultinomialNB')
