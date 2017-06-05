from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
import time, collections, numbers

def _print_time_in_miliseconds(start, end):
	if (type(start) != float and type(end) != float):
		print "Error when printing time: given start and end time must be float!"
		return
	duration = end - start
	addtional_info = " (%fs)" % duration if duration >= 10 else ""
	print "Time elapsed: %fms%s" % (duration * 1000, addtional_info)

_t0 = None

def tic():
    global _t0
    _t0 = time.time()

def toc():
    global _t0
    if _t0 == None:
        print "Please call tic() to start timer first"
        return

    _print_time_in_miliseconds(_t0, time.time())
    _t0 = None

#--------------------------------

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

def metric_analysis(y_test, y_predict, target_names, scores, clf_name):
    #Accuracy, recall, precision and confusion matrix
    print np.mean(y_predict == y_test)

    print metrics.classification_report(y_test, y_predict, target_names=target_names)

    print metrics.confusion_matrix(y_test, y_predict)

    #ROC
    n_classes = 2

    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test, scores[:, i])

    # Compute micro-average ROC curve and ROC area
    y_test_cat = []
    for i in range(n_classes):
        y_test_cat.append(y_test)
    y_test_cat = np.array(y_test_cat).T
    print scores.shape
    print y_test_cat.shape
    fpr['micro'], tpr['micro'], _ = metrics.roc_curve(y_test_cat.ravel(), scores.ravel())

    # Compute macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr

    # Plot ROC
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'], label='micro-average ROC', color='deeppink')
    plt.plot(fpr['macro'], tpr['macro'], label='macro-average ROC', color='aqua')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('ROC_%s' %clf_name)
#    colors = ['aqua', 'darkorange']
#    for i, color, label in zip(range(n_classes), colors, target_names):
#        plt.plot(fpr[i], tpr[i], color=color,
#                 label=label)
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('ROC')
#    plt.legend(loc="lower right")
#    plt.savefig('ROC_%s' %clf_name)


def metric_analysis_decfun(y_test, y_predict, target_names, scores, clf_name):
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
    plt.savefig('ROC_%s' %clf_name)

def get_accuracy(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise Exception("y_pred and y_true has different length!")

    correct_count = 0
    for idx, val_pred in enumerate(y_pred):
        if val_pred == y_true[idx]:
            correct_count += 1

    return float(correct_count) / len(y_pred)

def sort_matrix_diagonally(mat):
    if type(mat) != np.ndarray:
        raise Exception("sort_matrix_diagonally(): only accepts numpy array!")

    for ci in xrange(min(mat.shape)):
        # find row with maximum value on column ci
        maxval = 0
        maxidx = 0
        for idx, row in enumerate(mat):
            if idx >= ci and row[ci] > maxval:
                maxval = row[ci]
                maxidx = idx
        # swap row maxidx with row ci
        mat[ci], mat[maxidx] = mat[maxidx], mat[ci]

    return mat

def isnum(val):
    return isinstance(val, numbers.Number) or isinstance(val, np.number)

def is_legal_float(val):
    if isinstance(val, collections.Iterable):
        for x in val:
            if not is_legal_float(x):
                return False
        return True
    elif isnum(val):
        return abs(val) != float('inf') and val != float('nan')

def clamp(minval, val, maxval):
    if not (isnum(minval) and isnum(maxval)):
        return val

    if isinstance(val, collections.Iterable):
        for i, v in enumerate(val):
            val[i] = clamp(minval, v, maxval)
        return val
    elif isnum(val):
        if val > maxval:
            return maxval
        elif val < minval:
            return minval
        else:
            return val
    else:
        return val
