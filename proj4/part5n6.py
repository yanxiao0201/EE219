from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn import metrics
from sklearn.decomposition import *
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np
import random
import util
import scipy.sparse.linalg as ssl
#--------------------------------

### Loading and converting data

print "Loading comp and rec data..."
util.tic()

newsgroup = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

util.toc()

print "Vectorizing and converting to TFxIDF..."
util.tic()

count_vect = text.CountVectorizer(min_df=1, stop_words='english', analyzer = 'word', tokenizer=util.my_tokenizer)
tfidf_transformer = text.TfidfTransformer()

X_counts = count_vect.fit_transform(newsgroup.data)
X_tfidf = tfidf_transformer.fit_transform(X_counts)


print "Inspect singular values"
u,s,v = ssl.svds(X_tfidf, k=50)
print s

util.toc()

#--------------------------------

### parameters for this part

dims = [3,4,6,8,10]#add more dims here
#dim = 6
k_clses = [6,20]#add more groups here
#random_state = random.randint(1,100)
random_state = 33

#--------------------------------

### producing true classes
group20_true = newsgroup.target

group6_true = [0] * len(newsgroup.filenames)
classes = {
	0 : ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
	1 : ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
	2 : ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
	3 : ['misc.forsale'],
	4 : ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
	5 : ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']
}

for idx, fname in enumerate(newsgroup.filenames):
    class_name = fname.split('/')[-2]
    for cls in classes:
    	if class_name in classes[cls]:
    		group6_true[idx] = cls
    		break

#--------------------------------

### Machine learning

# reducing dimensions



normalizer_svd = Normalizer(copy=False)
normalizer_nmf = Normalizer(copy=False)



def cluster_analysis(dim,k_cls):
    X_dim_red = {}
    print "Reducing dimensions..."
    util.tic()
    svd_cls = TruncatedSVD(n_components=dim, algorithm='arpack')
    svd = make_pipeline(normalizer_svd,svd_cls)
    X_dim_red['SVD'] = svd.fit_transform(X_tfidf)

    nmf_cls = NMF(n_components=dim, init='random', random_state=random_state)
    nmf = make_pipeline(normalizer_nmf,nmf_cls)
    X_dim_red['NMF'] = nmf.fit_transform(X_tfidf)
    util.toc()

# transform
    #X_dim_red['SVD'] += np.max(X_dim_red['SVD'])
    #X_dim_red['SVD'] = X_dim_red['SVD'] ** 2
    X_dim_red['NMF'] = util.clamp(-10, np.log(X_dim_red['NMF']), 10)

# clustering
    print "Clustering..."
    util.tic()

    kmeans = {}
    kmeans['SVD'] = KMeans(n_clusters=k_cls, random_state=random_state).fit(X_dim_red['SVD'])
    kmeans['NMF'] = KMeans(n_clusters=k_cls, random_state=random_state).fit(X_dim_red['NMF'])

    util.toc()

#--------------------------------

### Evaluation

# Purity statistics

    if k_cls == 6:
        y_true = group6_true
    elif k_cls == 20:
        y_true = group20_true
    print "Purity stats report:"
    print "Dimension = %d" % dim
    print "No. of groups = {}".format(k_cls)
    for method in ['SVD', 'NMF']:
	   conf_mat = metrics.confusion_matrix(y_true, kmeans[method].labels_)
	   print "======== Method: %s ========" % method
	   print "Confusion Matrix:"
	   print conf_mat
	   #print "Confusion Matrix (w/ best permutation):"
	   #print util.sort_matrix_diagonally(conf_mat)
	   print "Homogeneity_score = {:4f}".format(homogeneity_score(y_true,kmeans[method].labels_))
	   print "Completeness_score = {:4f}".format(completeness_score(y_true,kmeans[method].labels_))
	   print "Adjusted_rand_score = {:4f}".format(adjusted_rand_score(y_true,kmeans[method].labels_))
	   print "Adjusted_mutual_info_score = {:4f}".format(adjusted_mutual_info_score(y_true,kmeans[method].labels_))

    return

if __name__ == '__main__':
	for dim in dims:
	    for k_cls in k_clses:
		cluster_analysis(dim,k_cls)
