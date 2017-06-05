from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp

# originally outside the function as global variables
_stemmer = SnowballStemmer("english")

_EXCLUDED_CHARS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

def my_tokenizer(s):
	# getting rid of numbers and symbols except defined excluded chars
	s = ''.join([ch for ch in s if ch.isalpha() or ch.isspace() or ch in _EXCLUDED_CHARS])
	# replace all excluded chars w/ space (may change)
	for ch in _EXCLUDED_CHARS:
		s = s.replace(ch, ' ')

	tokens = s.split()
	for idx, w in enumerate(tokens):
		tokens[idx] = _stemmer.stem(w)

	return tokens


def metric_analysis_decfun(y_test, y_predict, target_names, scores, clf_name, reduce_mtd):
    #Accuracy, recall, precision and confusion matrix
    print "Accuracy: {}".format(np.mean(y_predict == y_test))
    print ""
    print "Classification report:"
    print metrics.classification_report(y_test, y_predict, target_names=target_names)

    print "Confusion matrix:"
    print metrics.confusion_matrix(y_test, y_predict)

    #ROC

    fpr, tpr, _ = metrics.roc_curve(y_test, scores)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')

    print clf_name
    print reduce_mtd
    plt.savefig("ROC_{}_{}".format(clf_name,reduce_mtd))
