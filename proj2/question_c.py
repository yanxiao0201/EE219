from operator import itemgetter
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import time
import util

#--------------------------------

print "Loading data..."
t0 = time.time()

newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

util.print_time_in_miliseconds(t0, time.time())

#--------------------------------

target_name2idx = {}
for idx, target_name in enumerate(newsgroup_train.target_names):
	target_name2idx[target_name] = idx

#--------------------------------

print "Grouping data of each document into classes..."
t0 = time.time()

data_by_class = [""] * len(newsgroup_train.target_names)
for idx, data in enumerate(newsgroup_train.data):
	fname = newsgroup_train.filenames[idx]
	class_name = fname.split('/')[-2]
	class_idx = target_name2idx[class_name]

	data_by_class[class_idx] += data + "\n"

util.print_time_in_miliseconds(t0, time.time())

#--------------------------------

# vectorize
print "Vectorizing..."
t0 = time.time()

count_vect = text.CountVectorizer(min_df=1, stop_words='english', tokenizer=util.my_tokenizer)
Xc_train_counts = count_vect.fit_transform(data_by_class)

util.print_time_in_miliseconds(t0, time.time())

#--------------------------------

# TFxICF
tficf_transformer = text.TfidfTransformer()
X_train_tficf = tficf_transformer.fit_transform(Xc_train_counts)
print "Shape of TFxICF matrix: %s" % (X_train_tficf.shape,)

#--------------------------------

print ""
t0 = time.time()

# Finding the top 10 in certain class
reverse_vocab_dict = {}
for term in count_vect.vocabulary_:
	term_idx = count_vect.vocabulary_[term]
	reverse_vocab_dict[term_idx] = term

"""
# investigation
dictionary = {}
alphabet = 'abcdefghijklmnopqrstuvwxyz'
for ch in alphabet:
	dictionary[ch] = []
for term in count_vect.vocabulary_:
	start_letter = term[0]
	if start_letter in alphabet:
		dictionary[start_letter].append(term)
	else:
		print "Cannot add term to dictionary: %s" % term
"""

target_classes = ['comp.sys.ibm.pc.hardware' , 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
for class_name in target_classes:
	X_train_tficf_array = X_train_tficf.toarray()
	class_idx = target_name2idx[class_name]
	sig_arr = [(idx, val) for idx, val in enumerate(X_train_tficf_array[class_idx])]
	top10 = sorted(sig_arr, key=itemgetter(1), reverse=True)[:10]
	#print top10
	
	print "Top 10 significant terms in class %s:" % class_name
	for idx, val in enumerate(top10):
		term_idx, sig_val = val
		print "%-16s(significance = %f)" % (reverse_vocab_dict[term_idx], sig_val)

	print ""	# new line for every target class
