"""
Robert Harrison
Lucy Stuehrmann
Brady Snowden

tools.py
python file that contains the function which preprocesses the data
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras import utils as np_utils

def preprocess():
    # download the images from the .npy files
    images = np.load('images.npy')
    labels = np.load('labels.npy')
    # change the images matrix array to flattened vectors
    images = np.reshape(images, (len(labels), 784))

    # convert label numbers into one-hot encodings
    labels = np_utils.to_categorical(labels, num_classes=10)
    # use train_test_split to take stratified samples of the images and split
    # them into their training, test, and validation sets
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=.25, random_state=42)

    images_valid, images_train, labels_valid, labels_train = train_test_split(
        images_train, labels_train, test_size=.8, random_state=42)
    return images_train, images_test, images_valid, labels_train, labels_test, labels_valid

# streamline the process of making confusion matrices
def confusion(test, predict):
    confusion_test = [0] * len(test)
    confusion_predict = [0] * len(predict)

    for n in range(0, len(test)):
        confusion_test[n] = np.argmax(test[n])
        confusion_predict[n] = np.argmax(predict[n])
    # create a confusion matrix for the test and prediction parts
    return confusion_matrix(confusion_test, confusion_predict)

# visualize the first 3 misinterpreted values from the learning algorithm
def visualize(images_input, labels_input, predict_input):
	images = np.reshape(images_input, (len(labels_input), 28,28))
	labels = [0] * len(labels_input)
	predict = [0] * len(predict_input)
	for n in range(0, len(labels_input)):
		labels[n] = np.argmax(labels_input[n])
		predict[n] = np.argmax(predict_input[n])
	n = 0
	i = 0
	for n in range(0, len(labels)):
		if labels[n] != predict[n]:
			plt.imshow(images[n], interpolation='nearest', cmap='binary')
			print('Actual image value:') 
			print(labels[n])
			print('Predicted image value:')
			print(predict[n])
			plt.show()
			i += 1
		if i == 3:
			break
	return 0;

# function the displays a confusion matrix
def dispMatrix(matrix, title):
	figure, ax = plt.subplots()
	plt.ylabel('Predictions')
	plt.xlabel('Actual')
	plt.title(title)
	plt.xticks([0,1,2,3,4,5,6,7,8,9])
	plt.yticks([0,1,2,3,4,5,6,7,8,9])
	ax.matshow(matrix, cmap=plt.cm.Spectral)
	x = [0,1,2,3,4,5,6,7,8,9]
	for i in x:
		for j in x:
			c = matrix[j,i]
			ax.text(i,j,str(c), va='center', ha='center')
	
	plt.show()
	print(matrix)


# create hand engineered features which are the pixel intensity averages
# over 9 sets of 3 concurrent rows in the image matrix. 
def CustomFeat(images_input, length):
	# reshape images into a 28-by-28 matrices
	images = np.reshape(images_input, (length, 28,28))
	images_custom = []
	for i in images:
		mylist = []
		avg3Row = 0
		n = 0
		for x in i:
			for y in x:
				n += 1
				if n%3 == 0:
					mylist.append(avg3Row/84)
					avg3Row = 0;
				avg3Row += y
		images_custom.append(mylist)
	return images_custom


	"""
# create hand engineered features including: average pixel intensity, average number of pixels,
# longest number of pixels in a row, number of empty spaces
def deprecated CustomFeat(images_input, length):
	# reshape images into a 28-by-28 matrices
	images = np.reshape(images_input, (length, 28,28))
	images_custom = []
	for i in images:
		avgInten = 0			# average pixel intensity
		avgPix = 0				# average number of pixels in the image
		pixInRow = 0			# longest streak of pixels in a row
		inARow = 0				# used to help keep track of pixels in a row
		empty = 0				# keep track of # of empty spaces
		for x in i:
			for y in x:
				#for y in x:
				avgInten += y
				if y > 0:
					avgPix += 1
					inARow += 1
				else:
					inARow = 0
					empty += 1
				if inARow > pixInRow:
					pixInRow = inARow
		avgInten = avgInten/784
		avgPix = avgPix/784
		images_custom.append([avgInten, avgPix, pixInRow, empty])
	return images_custom
"""
