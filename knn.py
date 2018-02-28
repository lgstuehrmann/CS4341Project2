from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tools import preprocess, confusion, visualize
import matplotlib.pyplot as plt

images_train, images_test, images_valid, labels_train, labels_test, labels_valid = preprocess()

neigh = KNeighborsClassifier(n_neighbors=9, weights='distance')
neigh.fit(images_train, labels_train)
result = neigh.predict(images_test)
matrix = confusion(labels_test, result)
print(matrix)
print(accuracy_score(labels_test, result))

figure, ax = plt.subplots()
plt.ylabel('Predictions')
plt.xlabel('Actual')
plt.title('Confusion Matrix for KNearestNeighbor')
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.yticks([0,1,2,3,4,5,6,7,8,9])

ax.matshow(matrix, cmap=plt.cm.Spectral)
x = [0,1,2,3,4,5,6,7,8,9]
for i in x:
    for j in x:
        c = matrix[j,i]
        ax.text(i,j,str(c), va='center', ha='center')

plt.show()

visualize(images_test, labels_test, result)
