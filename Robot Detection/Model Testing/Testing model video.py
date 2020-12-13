import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import pickle
cap = cv2.VideoCapture('ar_tag.avi')
loaded=pickle.load(open("modelKNN.sav","rb"))

while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        height,width,dimension=frame.shape
    except:
        break
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,threshold = cv2.threshold(grayscale,100,255,cv2.THRESH_BINARY)
    resize=cv2.resize(threshold,(30,30))
    reshaped=resize.reshape(-1)
    prediction=loaded.predict(pd.DataFrame([reshaped]))
    if -1 not in prediction[0]:
        cv2.rectangle(frame,(prediction[0][0],prediction[0][2]),(prediction[0][1],prediction[0][3]),(255,0,0),2)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()
