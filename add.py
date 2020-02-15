import dlib
import numpy as np
import os
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detector = dlib.get_frontal_face_detector()
pose_predictor_68_point = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')

def whirldata_face_detectors(img, number_of_times_to_upsample=1):
 return face_detector(img, number_of_times_to_upsample)

def whirldata_face_encodings(face_image,num_jitters=1):
 face_locations = whirldata_face_detectors(face_image)
 if(len(face_locations) == 0 or len(face_locations)>1):
     return []
 else:
    face_location = face_locations[0]
    pose_predictor = pose_predictor_68_point
    predictor = pose_predictor(face_image, face_location) 
    return np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters))

count = 0
video_capture = cv2.VideoCapture('http://192.168.0.3:8080/video')
while True:
    ret, img = video_capture.read()
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.destroyAllWindows()
        break	
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
if(len(faces) == 0):
    print("No face detected")
else:
    for (x,y,w,h) in faces:
        crop_img = img[y:y+h,x:x+w]
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        enc = whirldata_face_encodings(rgb_small_frame)
        if(len(enc)!=0):
            print("Press C")
            while True:    
                cv2.imshow('img',crop_img)
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    break	
            print("Enter the name for the student")
            name = input()
            if os.path.exists("./Dataset/"+name):
                for each in os.listdir("./Dataset/"+name):
                    count = count+1 
            else:
                os.mkdir("./Dataset/"+name)
                count = 0
                np.savetxt("./Dataset/"+name+"/"+name+"_"+str(count)+".csv",enc,delimiter=",")
video_capture.release()
cv2.destroyAllWindows()