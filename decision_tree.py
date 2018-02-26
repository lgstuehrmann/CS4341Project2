"""
decision_tree.py
The Code for the decision tree model and training for 
project 2
"""
import numpy as np
from tools import preprocess
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
	min_samples_split=2, min_samples_lead=1, min_weight_fraction_leaf=0.0, max_features=None,
	random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
	class_weight=None, presort=False)

# train trees using fit
baseTree.fit(images_train, labels_train)
modTree.fit(images_train, labels_train)
# Now run an accuracy test of the tree
labels_predict = baseTree.predict(images_test)

accuracy_score(labels_test, labels_predict)
confusion_test1 = [0]*len(labels_test)
confusion_predict1 = [0]*len(labels_predict)
n = 0

for n in range(0, len(labels_test)):
	confusion_test1[n] = np.argmax(labels_test[n])
	confusion_predict1[n] = np.argmax(labels_predict[n])
# create a confusion matrix for the test and prediction parts
cm = confusion_matrix(confusion_test1, confusion_predict1)
print(cm)