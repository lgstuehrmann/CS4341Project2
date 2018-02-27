from keras import models
from kerasnn import x_test, y_test
from tools import confusion
import time
import numpy as np
import matplotlib.pyplot as plt

model = models.load_model('knn2000.h5')
print('Weights Loaded')
time.sleep(5)

predictions = model.predict(x_test, verbose=0)
matrix = confusion(y_test, predictions)

figure, ax = plt.subplots()
ax.matshow(matrix, cmap=plt.cm.Spectral)
x = [0,1,2,3,4,5,6,7,8,9]
for i in x:
    for j in x:
        c = matrix[j,i]
        ax.text(i,j,str(c), va='center', ha='center')

plt.show()
print(matrix)