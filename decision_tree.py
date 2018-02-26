"""
decision_tree.py
The Code for the decision tree model and training for 
project 2
"""
import numpy as np
from tools import preprocess
from tools import confusion
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


# load image data and labels data
images_train, images_test, images_valid, labels_train, labels_test, labels_valid = preprocess()
# create a decision tree using DecisionTreeClassifer
baseTree = DecisionTreeClassifier(random_state=100)
# create a modified tree changing different values to get 
# the confusion matrix to fit better
modTree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10,
	min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
	random_state=100, max_leaf_nodes=None, class_weight=None, presort=False)

# train trees using fit
baseTree.fit(images_train, labels_train)
modTree.fit(images_train, labels_train)
# Now create prediction values from the test values run on the tree
base_predict = baseTree.predict(images_test)
mod_predict = modTree.predict(images_test)

# create the confusion matrix for the trees
cm_base = confusion(labels_test, base_predict)
cm_mod = confusion(labels_test, base_predict)
print('Base Decision Tree Confusion Matrix')
print(cm_base)
print('Modified Decision Tree Confusion Matrix')
print(cm_mod)
print('Custom Feature Decision Tree Confusion Matrix')
"""
images_test = np.reshape(images_test, (28,28))
plt.imshow(images_test[1], interpolation='nearest')
plt.show()
"""