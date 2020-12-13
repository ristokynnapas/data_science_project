import numpy as np
import cv2
import time
robot=cv2.CascadeClassifier("1to1ratio100x100.xml")
cap = cv2.VideoCapture('ar_tag.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        height,width,dimension=frame.shape
        cop=frame.copy()
    except:
        break
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(grayscale,(5,5),0)
    robota=robot.detectMultiScale(blurred,1.3,1)
    for (x,y,w,h) in robota:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Robot",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(2)
cap.release()
cv2.destroyAllWindows()
