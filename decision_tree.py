"""
decision_tree.py
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import utils as np_utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from tools import preprocess
# load image data and labels data
images_train, images_test, images_valid, labels_train, labels_test, labels_valid = preprocess()

print(images_train[3])
print(labels_train[3])

