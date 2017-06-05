from common import *
import util
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

print "2-means..."

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_res = kmeans.fit_predict(X_tfidf)
#print kmeans_res.shape

class_label = [0] * len(newsgroup.filenames)

for idx, fname in enumerate(newsgroup.filenames):
    class_name = fname.split('/')[-2]
    if 'comp.' in class_name:
        class_label[idx] = 0
    elif 'rec.' in class_name:
        class_label[idx] = 1

confusion_res = confusion_matrix(class_label,kmeans_res,labels = [0,1])
print confusion_res
#print class_label.count(1)
#print list(kmeans_res).count(1)

print "Homogeneity_score = {:4f}".format(homogeneity_score(class_label,kmeans_res))
print "Completeness_score = {:4f}".format(completeness_score(class_label,kmeans_res))
print "Adjusted_rand_score = {:4f}".format(adjusted_rand_score(class_label,kmeans_res))
print "Adjusted_mutual_info_score = {:4f}".format(adjusted_mutual_info_score(class_label,kmeans_res))
