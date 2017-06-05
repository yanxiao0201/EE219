from part6 import *
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import util_2
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF
from sklearn import linear_model

#Vectorizer, TFxIDF and LSI
count_vect = text.CountVectorizer(min_df=1, stop_words='english', analyzer = 'word', tokenizer=util_2.my_tokenizer)

tfidf_transformer = text.TfidfTransformer()
svd = TruncatedSVD(n_components=50, algorithm='arpack')
nmf = NMF(n_components=50)

print "Vectorizing and converting to TFxIDF... (training)"

tweet_counts = count_vect.fit_transform(tweet_text)
tweet_tfidf = tfidf_transformer.fit_transform(tweet_counts)


target_names = ["Washington","Massachusetts"]


print "Reduce dimension by nmf..."
tweet_nmf = nmf.fit_transform(tweet_tfidf)
#print tweet_nmf.shape

#train_test_split data
X_train, X_test, y_train, y_test = train_test_split(tweet_nmf,tweet_addr,train_size=0.9, random_state=42)


####SVM####
print "SVM 2-groups...NMF"
clf = svm.LinearSVC(C=1e5)
clf.fit(X_train,y_train)
svm_predict = clf.predict(X_test)
svm_score = clf.decision_function(X_test)
util_2.metric_analysis_decfun(y_test, svm_predict, target_names, svm_score,clf_name = 'SVM', reduce_mtd = 'NMF')

####naive Bayes####
print "Naive Bayes...NMF"
clf2 = MultinomialNB()
clf2.fit(X_train, y_train) # train
nb_predict = clf2.predict(X_test)
nb_scores = clf2.predict_proba(X_test)
nb_pos_score = np.array([score[1] for score in nb_scores])
util_2.metric_analysis_decfun(y_test, nb_predict, target_names, nb_pos_score, 'Naive Bayes','NMF')

####logistic regression####
print "Logistic regression with regularization...NMF"
clf3 = linear_model.LogisticRegression(penalty = 'l1')
clf3.fit(X_train,y_train)
logreg_predict = clf3.predict(X_test)
logreg_score = clf3.predict_proba(X_test)
logreg_pos_score = np.array([score[1] for score in logreg_score])
util_2.metric_analysis_decfun(y_test, logreg_predict, target_names, logreg_pos_score, 'Logistic Regression','NMF')

print "Reduce dimension by SVD..."
tweet_svd = svd.fit_transform(tweet_tfidf)
#train_test_split data
X_train, X_test, y_train, y_test = train_test_split(tweet_svd,tweet_addr,train_size=0.9, random_state=42)


####SVM####
print "SVM 2-groups...SVD"
clf = svm.LinearSVC(C=1e5)
clf.fit(X_train,y_train)
svm_predict = clf.predict(X_test)
svm_score = clf.decision_function(X_test)
util_2.metric_analysis_decfun(y_test, svm_predict, target_names, svm_score, 'SVM','SVD')

####logistic regression####
print "Logistic regression with regularization...SVD"
clf3 = linear_model.LogisticRegression(penalty = 'l1')
clf3.fit(X_train,y_train)
logreg_predict = clf3.predict(X_test)
logreg_score = clf3.predict_proba(X_test)
logreg_pos_score = np.array([score[1] for score in logreg_score])
util_2.metric_analysis_decfun(y_test, logreg_predict, target_names, logreg_pos_score, 'Logistic Regression','SVD')
