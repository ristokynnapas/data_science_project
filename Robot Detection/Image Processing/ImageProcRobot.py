import numpy as np
import cv2
import time

def ImageProcess(image):
    blurred=cv2.GaussianBlur(image,(7,7),1)
    grayscale=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(grayscale,100,255,cv2.THRESH_BINARY)
    thresh = cv2.Canny(grayscale,50,255)
    kernel=np.ones((5,5))
    dilated=cv2.dilate(thresh,kernel,iterations=1)
    return dilated
def Robot(image, origImage):
    contours,hierarchy=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    posx=[]
    posy=[]
    for contour in contours:
        arc = cv2.arcLength(contour, True)
        area=cv2.contourArea(contour)
        perimeter=cv2.arcLength(contour, True)
        approximate=cv2.approxPolyDP(contour,0.02*perimeter,True)
        if arc>200 and arc<1500 and perimeter>100 and area>3700:
            x,y,w,h=cv2.boundingRect(approximate)
            cv2.rectangle(origImage,(x,y),(x+w,y+h),(255,255,0),3)
            centralx=(x+x+w)/2
            centraly=(y+y+h)/2
            posx.append(centralx)
            posy.append(centraly)
    if len(posx)>0:
        return (round(np.mean(posx)),round(np.mean(posy)))
    else:
        return None 
        #time.sleep(1)
cap = cv2.VideoCapture('ar_tag.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        height,width,dimension=frame.shape
    except:
        break
    
    dilated=ImageProcess(frame)
    Robot(dilated,frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
