import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups

newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
target_names_train = newsgroup_train.target_names

num_array = []
xs = []
num_comp = 0
num_rec = 0

for idx, target_name in enumerate(target_names_train):
    #num = len([fname for fname in newsgroup_train.filenames if target_name in fname])
    num = np.count_nonzero(newsgroup_train.target == idx)
    num_array.append(num)
    xs.append(idx + 1)
    if 'comp' in target_name:
        num_comp += num
    elif 'rec'in target_name:
        num_rec += num

print "Report:"
print "Computer Technology: %d\nRecreational Activity: %d" % (num_comp, num_rec)

if sum(num_array) != len(newsgroup_train.filenames):
    raise Exception('Data not recorded correctly.')

plt.xlim(0, 22)
plt.bar(xs, num_array)
plt.xticks([x + .5 for x in xs], target_names_train, rotation='vertical')
plt.savefig('qa_histogram.png', bbox_inches='tight')
#plt.show()
