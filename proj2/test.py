from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import time
import util

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
newsgroup_train = fetch_20newsgroups(subset='train', categories = categories, shuffle=True, random_state=42)
newsgroup_test = fetch_20newsgroups(subset='test', categories = categories, shuffle=True, random_state=42)
