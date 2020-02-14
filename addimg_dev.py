import dlib
import numpy as np
import os
import imageio

face_detector = dlib.get_frontal_face_detector()
pose_predictor_68_point = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')

def whirldata_face_detectors(img, number_of_times_to_upsample=1):
 return face_detector(img, number_of_times_to_upsample)

def whirldata_face_encodings(face_image,num_jitters=1):
 face_locations = whirldata_face_detectors(face_image)
 if(len(face_locations) == 0 or len(face_locations)>1):
     print("No faces or more than one faces")
     return []
 else:
    face_location = face_locations[0]
    pose_predictor = pose_predictor_68_point
    predictor = pose_predictor(face_image, face_location) 
    return np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters))

count = 0
img = imageio.imread("./gow.jpeg")
enc = whirldata_face_encodings(img)
if(len(enc)!=0):
    print("Enter the name for the student")
    name = input()
    if os.path.exists("./Dataset/"+name):
        for each in os.listdir("./Dataset/"+name):
            count = count+1 
    else:
        os.mkdir("./Dataset/"+name)
        count = 0
    np.savetxt("./Dataset/"+name+"/"+name+"_"+str(count)+".csv",enc,delimiter=",")