import pickle
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np

model_dict=pickle.load(open('./model.p','rb'))
model=model_dict['model']

base_options=python.BaseOptions(model_asset_path="hand_landmarker.task")
options=vision.HandLandmarkerOptions(base_options,num_hands=1)

detector=vision.HandLandmarker.create_from_options(options)

cap=cv2.VideoCapture(0)

labels_dict={0:'A',1:'B',2:'C',3:'D'}

while True:
    features=[]
    x_=[]
    y_=[]
    
    ret,frame=cap.read()
    
    if not ret:
        break

    H,W,_=frame.shape
    
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=frame_rgb)
    
    result=detector.detect(mp_image)
    
    if result.hand_landmarks:
        hand_landmarks=result.hand_landmarks[0]
        
        for landmarks in hand_landmarks:
            x_.append(landmarks.x)
            y_.append(landmarks.y)
            
        for landmark in hand_landmarks:
            features.append(landmark.x-min(x_))
            features.append(landmarks.y-min(y_))
            
        x1=int(min(x_)*W)-10
        y1=int(min(y_)*H)-10
        
        x2=int(min(x_)*W)+10
        y2=int(min(y_)*H)+10
        
        prediction=model.predict([np.asarray(features)])
        predicted_character=labels_dict[int(prediction[0])]
        
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,predicted_character,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,0),3,cv2.LINE_AA)
        
        cv2.imshow('frame',frame)
        key=cv2.waitKey(1)
        
        if key==ord('a'):
            break
        
cap.release()
cv2.destoryAllWindows()



