"""
tools.py
python file that contains the function which preprocesses the data
"""
from keras import utils as np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
def preprocess():
	# download the images from the .npy files
	images = np.load('images.npy')
	labels = np.load('labels.npy')
	# change the images matrix array to flattened vectors
	images = np.reshape(images, (len(labels), 784))

	# convert label numbers into one-hot encodings
	labels = np_utils.to_categorical(labels, num_classes = 10)
	# use train_test_split to take stratified samples of the images and split
	# them into their training, test, and validation sets 
	images_train, images_test, labels_train, labels_test = train_test_split(
			images, labels, test_size = .25, random_state = 42)

	images_valid, images_train, labels_valid, labels_train = train_test_split(
			images_train, labels_train, test_size = .8, random_state = 42)
	return images_train, images_test, images_valid, labels_train, labels_test, labels_valid

# streamline the process of making confusion matrices
def confusion(test, predict):
	confusion_test = [0]*len(test)
	confusion_predict = [0]*len(predict)

	for n in range(0, len(test)):
		confusion_test[n] = np.argmax(test[n])
		confusion_predict[n] = np.argmax(predict[n])
	# create a confusion matrix for the test and prediction parts
	return confusion_matrix(confusion_test, confusion_predict)