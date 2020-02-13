import os
import numpy as np
import math
from sklearn import neighbors
import pickle
x = []
y = []
for student in os.listdir('./Dataset'):
    for img in os.listdir('./Dataset/'+student):
        x.append(np.genfromtxt('./Dataset/'+student+"/"+img,delimiter = ','))
        y.append(student)
n_neighbors = int(round(math.sqrt(len(x))))
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
knn_clf.fit(x, y)
knnPickle = open('model.sav', 'wb') 
pickle.dump(knn_clf, knnPickle)