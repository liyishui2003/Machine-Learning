from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import numpy as np

def knn_fit(X_train, y_train):
    #knn does not require training on the data
    #So..do nothing,just return
    return X_train, y_train
def knn_predict(X_train, y_train, X_test, k):
    #calculate distance by Euclidean Distance
    distances = np.sqrt(np.sum((X_train - X_test)**2, axis=1))
    #sort and fetch the first K
    k_neighbors = np.argsort(distances)[:k]
    #cal the lable that appears most
    class_counts = np.bincount(y_train[k_neighbors])
    return np.argmax(class_counts)
'''
data = sklearn.datasets.iris.data
label = sklearn.datasets.iris.target..不知道为啥用不了
'''
iris = datasets.load_iris() # import dataset
data = iris.data
target = iris.target
gap = LeaveOneOut() # "留一法"
K = []
Accuracy = []

for k in range(1, 10):
    correct=0
    for train,test in gap.split(data):
        X_train, X_test = data[train], data[test]
        y_train, y_test = target[train], target[test]
        X_train, y_train = knn_fit(X_train, y_train)  
        y_sample = knn_predict(X_train, y_train, X_test, k)
        if y_sample == y_test:
            correct += 1
    Accuracy.append(correct/len(data))
    K.append(k)

    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# knn_by_me
ax1.plot(K, Accuracy)
ax1.set_title('KNN_by_liyishui')
ax1.set_xlabel('K')
ax1.set_ylabel('Accuracy')


K2 = []
Ac2 = []
for k in range(1, 10):
    correct = 0
    knn = KNeighborsClassifier(k)

    for train, test in gap.split(data):
        knn.fit(data[train], target[train])
        y_sample = knn.predict(data[test]) 
        if y_sample == target[test]:
            correct += 1
    Ac2.append(correct/len(data))
    K2.append(k)

# knn_by_sklearn
ax2.plot(K, Accuracy)
ax2.set_title('KNN_by_sklearn')
ax2.set_xlabel('K2')
ax2.set_ylabel('Accuracy2')
plt.tight_layout()
