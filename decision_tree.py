"""
decision_tree.py
The Code for the decision tree model and training for 
project 2
"""
import numpy as np
import tools as tools
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras import utils as np_utils
target_names = ['0','1','2','3','4','5','6','7','8','9']

# load image data and labels data
images_train, images_test, images_valid, labels_train, labels_test, labels_valid = tools.preprocess()

#create the Custom Features for each of the sets
cimg_train = tools.CustomFeat(images_train, len(labels_train))
cimg_valid = tools.CustomFeat(images_valid, len(labels_valid))
cimg_test = tools.CustomFeat(images_test, len(labels_test))

# Start by Creating the 3 Decision Tree models
# Create a basic decision tree using DecisionTreeClassifer
basTree = DecisionTreeClassifier(random_state=100)
# Create a modified tree changing different values to get 
# the confusion matrix to fit better
modTree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20,
	min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
	random_state=100, max_leaf_nodes=None, class_weight=None, presort=False)
# Create a Decision Tree based on custom arguements made
# from the MNIST data
custTree = DecisionTreeClassifier(random_state=100)

# train trees using fit
basTree.fit(images_train, labels_train)
modTree.fit(images_train, labels_train)
custTree.fit(cimg_train, labels_train)

# check accuracy of Decision Trees on validation set
base_val = basTree.predict(images_valid)
mod_val = modTree.predict(images_valid)
cust_val = custTree.predict(cimg_valid)

print('Basic Tree Validation Accuracy')
print(accuracy_score(labels_valid, base_val))
print('Modified Tree Validation Accuracy')
print(accuracy_score(labels_valid, mod_val))
print('Custom Tree Validation Accuracy')
print(accuracy_score(labels_valid, cust_val))

# Now create prediction values from the test values run on the tree
bas_predict = basTree.predict(images_test)
mod_predict = modTree.predict(images_test)
cust_predict = custTree.predict(cimg_test)


# create the confusion matrix for the base tree
cm_bas = tools.confusion(labels_test, bas_predict)

print('Classification Report for Basic Tree')
print('Basic Tree Test Accuracy')
print(accuracy_score(labels_test, bas_predict))
print(classification_report(labels_test, bas_predict))
tools.dispMatrix(cm_bas, 'Base Decision Tree Confusion Matrix')

# Look at visualizations for base decision tree
print('Visualization of 3 mistakes made in base tree')
tools.visualize(images_test, labels_test, bas_predict)

# create the confusion matrix for the modified tree
cm_mod = tools.confusion(labels_test, mod_predict)
print('Classification Report for Modified Tree')
print('Modified Tree Test Accuracy')
print(accuracy_score(labels_test, mod_predict))
print(classification_report(labels_test, mod_predict))
tools.dispMatrix(cm_mod, 'Modified Decision Tree Confusion Matrix')
# Look at the visualizations for the modified decision tree
print('Visualization of 3 mistakes made in modified tree')
tools.visualize(images_test, labels_test, mod_predict)

# create the confusion matrix for the Custom Features Tree
cm_cust = tools.confusion(labels_test, cust_predict)
print(cm_cust)
print('Classification Report for Custom Feature Tree')
print('Custom Feature Tree Test Accuracy')
print(accuracy_score(labels_test, cust_predict))
print(classification_report(labels_test, cust_predict))
tools.dispMatrix(cm_cust, 'Custom Feature Decision Tree Confusion Matrix')
# Look at the visualizations for the custom Features Tree
print('Visualization of 3 mistakes made in Custom Features tree')
tools.visualize(images_test, labels_test, cust_predict)