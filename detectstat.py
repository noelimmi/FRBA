import os
import dlib
import numpy as np
import math
from sklearn import neighbors
import imageio
import pickle


face_detector = dlib.get_frontal_face_detector()
pose_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')
distance_threshold = 0.6
knn = pickle.load(open('model.sav','rb'))
img = imageio.imread('./two.jpg')
face_locations = face_detector(img,0)
for face_location in face_locations:
    predictor = pose_predictor(img, face_location)
    encoding = np.array(face_encoder.compute_face_descriptor(img, predictor, 1))
    closest_distances = knn.kneighbors(encoding.reshape(1,-1),n_neighbors = 1)
    if(closest_distances[0][0][0]>=0.6):
        print("unknown")
    else:
        print(knn.predict((np.array(face_encoder.compute_face_descriptor(img, predictor, 1))).reshape(1,-1))[0]) 