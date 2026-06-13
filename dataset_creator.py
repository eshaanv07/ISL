import os
import pickle
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

base_options=python.BaseOptions(model_asset_path="hand_landmarker.task")
options=vision.HandLandmarkerOptions(base_options=base_options,num_hands=1)

detector=vision.HandLandmarker.create_from_options(options)

DATASET_DIR='./Indian'

data=[]
labels=[]

for dir in os.listdir(DATASET_DIR):
    for img_path in os.listdir(os.path.join(DATASET_DIR,dir)):
        full_path=os.path.join(DATASET_DIR,dir,img_path)
        image=mp.Image.create_from_file(full_path)
        result=detector.detect(image)
        if result.hand_landmarks:
            hand=result.hand_landmarks[0]
            
            x_=[]
            y_=[]
            
            for landmark in hand:
                x_.append(landmark.x)
                y_.append(landmark.y)
                
            features=[]
            
            for landmark in hand:
                features.append(landmark.x-min(x_))
                features.append(landmark.y-min(y_))
                
            data.append(features)
            labels.append(dir)
            
with open("data.pickle","wb") as f:
    pickle.dump({"data":data,"labels":labels},f)
            