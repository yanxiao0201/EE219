from sklearn.decomposition import *
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from common import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import util
import scipy.sparse.linalg as ssl

y_true = [0 if x <= 3 else 1 for x in newsgroup.target]

# parameters
random_state = 42
visualize = True
transform_method = 'none'
normalize = False
#dims = [2,3,4,5,6,7,8,9,10]
dims = [2]
showtrue = False
colorcode = True

perf_stats = {}
draw_perf_stats = True

# argument reading
if len(sys.argv) > 1:
	for argv in sys.argv[1:]:
		# splitting argument
		argv = argv.split('=')
		opt = argv[0]
		val = None
		if len(argv) > 1:
			val = argv[1]

		# reading argument
		if opt == '--normalize':
			normalize = True
		elif opt == '--novis':
			visualize = False
		elif opt == '--nostats':
			draw_perf_stats = False
		elif opt == '--showtrue':
			showtrue = True
		elif opt == '--nocolor':
			colorcode = False
		elif val != None:
			if opt == '--transform':
				transform_method = val
			elif opt == '--dims':
				# very insecure!!!
				val = eval(val)
				if type(val) == list:
					# can be still very insecure
					dims = [int(x) for x in val]
			elif opt == '--random_state':
				random_state = int(val)

def cluster_n_analyze(dim=2):
	print "Dimension = %d" % dim

	# reducing dimensions
	print "Reducing dimensions..."
	util.tic()

	normalizer = Normalizer(copy=False)
	svd_cls = TruncatedSVD(n_components=dim ,n_iter = 10, random_state= random_state)
	nmf_cls = NMF(n_components=dim, init='random', random_state=random_state)

	X_dim_red = {}

	#svd = make_pipeline(svd_cls, normalizer) if normalize else svd_cls
	svd = make_pipeline(normalizer,svd_cls) if normalize else svd_cls
	X_dim_red['SVD'] = svd.fit_transform(X_tfidf)
	print svd_cls.explained_variance_ratio_

	#nmf = make_pipeline(nmf_cls, normalizer) if normalize else nmf_cls
	nmf = make_pipeline(normalizer,nmf_cls) if normalize else nmf_cls
	X_dim_red['NMF'] = nmf.fit_transform(X_tfidf)

	#print X_dim_red['SVD'][0]

	util.toc()

	# apply non-linear transformation here
	if transform_method == 'exp':
		X_dim_red['SVD'] = np.exp(X_dim_red['SVD'])
		X_dim_red['NMF'] = np.exp(X_dim_red['NMF'])
	elif transform_method == 'sqrt':
		X_dim_red['NMF'] = np.sqrt(X_dim_red['NMF'])
	elif transform_method == 'log':
		X_dim_red['NMF'] = util.clamp(-10, np.log(X_dim_red['NMF']), 10)
	elif transform_method == 'customized':
		# different transformation to SVD and NMF
		X_dim_red['SVD'] += np.max(X_dim_red['SVD'])
		X_dim_red['SVD'] = X_dim_red['SVD'] ** 2
		X_dim_red['NMF'] = util.clamp(-10, np.log(X_dim_red['NMF']), 10)

	# clustering
	#print X_dim_red['SVD'][0]
	print "Clustering..."
	util.tic()

	kmeans = {}
	kmeans['SVD'] = KMeans(n_clusters=2, random_state=random_state).fit(X_dim_red['SVD'])
	kmeans['NMF'] = KMeans(n_clusters=2, random_state=random_state).fit(X_dim_red['NMF'])

	util.toc()

	# Purity statistics
	print "Purity stats report:"
	print "Dimension = %d" % dim
	for method in ['SVD', 'NMF']:
		scores = []
		scores.append(homogeneity_score(y_true,kmeans[method].labels_))
		scores.append(completeness_score(y_true,kmeans[method].labels_))
		scores.append(adjusted_rand_score(y_true,kmeans[method].labels_))
		scores.append(adjusted_mutual_info_score(y_true,kmeans[method].labels_))

		# document statistics
		if method not in perf_stats:
			perf_stats[method] = [[], [], [], []]

		for idx, arr in enumerate(perf_stats[method]):
			arr.append(scores[idx])
			perf_stats[method][idx] = arr

		# print...
		print "======== Method: %s ========" % method
		print "Confusion Matrix:"
		print metrics.confusion_matrix(y_true, kmeans[method].labels_)
		print "Homogeneity_score = {:4f}".format(scores[0])
		print "Completeness_score = {:4f}".format(scores[1])
		print "Adjusted_rand_score = {:4f}".format(scores[2])
		print "Adjusted_mutual_info_score = {:4f}".format(scores[3])

	# stop if no visualization is needed
	if not visualize:
		return

	# Visualization
	for method in ['SVD', 'NMF']:
		print method

		xs, ys = [[],[]], [[],[]]
		xt, yt = [[],[]], [[],[]]
		xe0, ye0 = [], []
		xe1, ye1 = [], []

		labels = kmeans[method].labels_
		# switch group if confusion matrix proved it to be a better match
		conf_mat = metrics.confusion_matrix(y_true, kmeans[method].labels_)
		if conf_mat[0][0] + conf_mat[1][1] < conf_mat[0][1] + conf_mat[1][0]:
			# since it only contains 0 and 1...
			labels = [1 - x for x in labels]

		for idx, val in enumerate(X_dim_red[method]):
			# projection: may find some other method
			xval = val[0]
			yval = val[1]

			label = labels[idx]
			truth = y_true[idx]

			if label == 0 and truth == 1:
				# errorneously put into category 0
				xe0.append(xval)
				ye0.append(yval)
			elif label == 1 and truth == 0:
				# errorneously put into category 1
				xe1.append(xval)
				ye1.append(yval)

			xs[label].append(xval)
			ys[label].append(yval)

			xt[truth].append(xval)
			yt[truth].append(yval)

		#if method == 'NMF':
		#	plt.xscale('log')
		#	plt.yscale('log')

		plt.figure()
		plt.scatter(xs[0], ys[0], c='r' if colorcode else 'k', marker=',', lw=0, s=1)
		plt.scatter(xs[1], ys[1], c='b' if colorcode else 'k', marker=',', lw=0, s=1)
		if not showtrue and colorcode:
			plt.scatter(xe0, ye0, c='g', marker=',', lw=0, s=1)
			plt.scatter(xe1, ye1, c='y', marker=',', lw=0, s=1)

		optional_str = ": kmeans\n" if showtrue else ""
		plt.title("%s%s" % (method, optional_str))
		transf_opt = 'pre_transf' if transform_method == 'none' else 'post_transf'
		plt.savefig('scatter_%s_dim%d_%s.png' % (method, dim, transf_opt))
		plt.show()

		if showtrue:
			plt.figure()
			plt.scatter(xt[0], yt[0], c='r', marker=',', lw=0, s=1)
			plt.scatter(xt[1], yt[1], c='b', marker=',', lw=0, s=1)

			plt.title("%s: true\ndim = %d" % (method, dim))
			plt.show()


#if __name__ == '__main__':
# main

print "Inspect singular values"
k=100
u,s,v = ssl.svds(X_tfidf, k=k)
print s

#x = range(len(s), 0, -1)
#plt.plot(x,s)
#plt.show()

for dim in dims:
	cluster_n_analyze(dim)

# plot performance stats
if len(dims) > 1 and draw_perf_stats:
	for method in perf_stats:
		lines = []
		for arr in perf_stats[method]:
			line, = plt.plot(dims, arr)
			lines.append(line)
		plt.legend(lines, ['homogeneity', 'completeness', 'adjusted_rand', 'adjusted_mutual_info'])
		plt.title("%s Performance" % method)
		plt.show()
