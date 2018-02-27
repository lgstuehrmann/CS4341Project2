from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from tools import preprocess, confusion
import numpy as np
import matplotlib.pyplot as plt

images_train, images_test, images_valid, labels_train, labels_test, labels_valid = preprocess()

neighbors = list(range(1, 26, 2))

ac_scores_uniform = []
ac_scores_distance = []
weight = 'uniform'
algo = 'auto'
leaf = 30
p = 2

# for k in neighbors:
#     # weights uniform, distance
#     # algorithm auto, ball_tree, kd_tree, brute
#     neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', leaf_size=leaf, p=p,
#                                  metric='minkowski', metric_params=None, n_jobs=1)
#     neigh.fit(images_train, labels_train)
#
#     result = neigh.predict(images_test)
#     ac_scores_uniform.append(accuracy_score(labels_test, result))
#     print(k)
#
# for k in neighbors:
#     # weights uniform, distance
#     # algorithm auto, ball_tree, kd_tree, brute
#     neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', leaf_size=leaf, p=p,
#                                  metric='minkowski', metric_params=None, n_jobs=1)
#     neigh.fit(images_train, labels_train)
#
#     result = neigh.predict(images_test)
#     ac_scores_distance.append(accuracy_score(labels_test, result))
#     print(k)
neigh = KNeighborsClassifier(n_neighbors=10, weights=weight, algorithm=algo, leaf_size=leaf, p=p, metric='minkowski', metric_params=None, n_jobs=1)
neigh.fit(images_train, labels_train)
result = neigh.predict(images_test)
matrix = confusion(labels_test, result)
print(matrix)
print(accuracy_score(labels_test, result))

figure, ax = plt.subplots()
plt.ylabel('Predictions')
plt.xlabel('Actual')
plt.title('Confusion Matrix for Keras Neural Net')
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.yticks([0,1,2,3,4,5,6,7,8,9])

ax.matshow(matrix, cmap=plt.cm.Spectral)
x = [0,1,2,3,4,5,6,7,8,9]
for i in x:
    for j in x:
        c = matrix[j,i]
        ax.text(i,j,str(c), va='center', ha='center')

plt.show()

# plt.plot(neighbors, ac_scores_uniform, 'b-', label='Uniform')
# plt.plot(neighbors, ac_scores_distance, 'r-', label='Distance')
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Accuracy')
# plt.title('K value vs Accuracy for Different Weights')
# plt.legend()
# plt.show()

# [[152   1   0   0   0   0   0   0   0   0]
#  [  0 177   4   0   0   0   0   1   0   1]
#  [  0   3 168   1   0   0   1   3   2   0]
#  [  0   1   0 150   1   7   3   2   2   2]
#  [  0   4   0   0 127   0   1   0   0  13]
#  [  0   1   0   2   2 126   4   0   0   1]
#  [  1   2   0   0   0   1 150   0   1   0]
#  [  1   5   0   1   1   0   0 169   0   6]
#  [  0   4   2   3   0   7   3   0 143   1]
#  [  1   1   0   0   5   0   0   6   0 148]]

# neighbors = list(range(1, 10, 2))
#
# # empty list that will hold cv scores
# cv_scores = []
#
# # perform 10-fold cross validation
# for k in neighbors:
#     knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
#     knn.fit(images_train, labels_train)
#     scores = cross_val_score(knn, images_valid, labels_valid, cv=KFold(shuffle=True), scoring='accuracy')
#     cv_scores.append(scores.mean())
#     print(confusion())
#     print(k)
#
# # changing to misclassification error
# MSE = [1 - x for x in cv_scores]
#
# # determining best k
# optimal_k = neighbors[MSE.index(min(MSE))]
# print("The optimal number of neighbors is %d" % optimal_k)
#
# # plot misclassification error vs k
# plt.plot(neighbors, MSE)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Misclassification Error')
# plt.show()
