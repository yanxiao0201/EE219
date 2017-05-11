from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import time
import util

print "Loading data..."
t0 = time.time()

newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

util.print_time_in_miliseconds(t0, time.time())

print "Vectorizing..."
t0 = time.time()

#tokenize
count_vect = text.CountVectorizer(min_df=1, stop_words='english', tokenizer=util.my_tokenizer)
X_train_counts = count_vect.fit_transform(newsgroup_train.data)
print X_train_counts.shape

#print count_vect.vocabulary_.keys()[0:200]

util.print_time_in_miliseconds(t0, time.time())

#TFxIDF
tfidf_transformer = text.TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print X_train_tfidf.shape
