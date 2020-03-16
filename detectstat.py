import os
import dlib
import numpy as np
import math
from sklearn import neighbors
import cv2
import pickle
import sqlite3
import datetime

detected = []
face_detector = dlib.get_frontal_face_detector()
pose_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')
distance_threshold = 0.6
knn = pickle.load(open('model.sav','rb'))
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
         break	
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
img = small_frame[:, :, ::-1]
face_locations = face_detector(img,1)
new = datetime.datetime.now().strftime("%m.%d.%Y. %H:%M:%S")
conn = sqlite3.connect('./Database/'+new+'.db')
c = conn.cursor()
c.execute('''CREATE TABLE record (student text, presence text)''')
default = "No"
for student in os.listdir("./Dataset"):
    c.execute("insert into record (student,presence) values (?, ?)",(student,default))
for face_location in face_locations:
    predictor = pose_predictor(img, face_location)
    encoding = np.array(face_encoder.compute_face_descriptor(img, predictor, 1))
    closest_distances = knn.kneighbors(encoding.reshape(1,-1),n_neighbors = 1)
    if(closest_distances[0][0][0]>=0.5):
        pass
    else:
        predicted = knn.predict((np.array(face_encoder.compute_face_descriptor(img, predictor, 1))).reshape(1,-1))[0]
        if(predicted not in detected):
            detected.append(predicted)
            print(closest_distances[0][0][0])
            print(predicted)
            c.execute('UPDATE record SET presence = ? WHERE student = ? ',('Yes',predicted))
conn.commit()