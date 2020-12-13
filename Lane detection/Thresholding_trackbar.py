import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('./test_images/figure0.jpg') # insert image

window = 'HSV trackbar thresholding'
cv2.namedWindow(window)


#    _____        __            _ _     _   _                   _           _     _     
#   |  __ \      / _|          | | |   | | | |                 | |         | |   | |    
#   | |  | | ___| |_ __ _ _   _| | |_  | |_| |__  _ __ ___  ___| |__   ___ | | __| |___ 
#   | |  | |/ _ \  _/ _` | | | | | __| | __| '_ \| '__/ _ \/ __| '_ \ / _ \| |/ _` / __|
#   | |__| |  __/ || (_| | |_| | | |_  | |_| | | | | |  __/\__ \ | | | (_) | | (_| \__ \
#   |_____/ \___|_| \__,_|\__,_|_|\__|  \__|_| |_|_|  \___||___/_| |_|\___/|_|\__,_|___/
#                                                                                       
#                                                                                       


high_H = 255
high_S = 255
high_V = 255
low_H = 25
low_S = 25
low_V = 25

track_high_H = 'High Hue'
track_high_S = 'High Saturation'
track_high_V = 'High Value'
track_low_H = 'Low Hue'
track_low_S = 'Low Saturation'
track_low_V = 'Low Value'


#    _______             _    _                   __                  _   _                 
#   |__   __|           | |  | |                 / _|                | | (_)                
#      | |_ __ __ _  ___| | _| |__   __ _ _ __  | |_ _   _ _ __   ___| |_ _  ___  _ __  ___ 
#      | | '__/ _` |/ __| |/ / '_ \ / _` | '__| |  _| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#      | | | | (_| | (__|   <| |_) | (_| | |    | | | |_| | | | | (__| |_| | (_) | | | \__ \
#      |_|_|  \__,_|\___|_|\_\_.__/ \__,_|_|    |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#                                                                                           
#                                                                                           



def high_H_trackbar(val):
    global high_H
    global low_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(track_high_H , window, high_H)
def high_S_trackbar(val):
    global high_S
    global low_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(track_high_S, window, high_S)
def high_V_trackbar(val):
    global high_V
    global low_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(track_high_V, window, high_V)

def low_H_trackbar(val):
    global high_H
    global low_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(track_low_H , window, low_H)
def low_S_trackbar(val):
    global high_S
    global low_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(track_low_S, window, low_S)
def low_V_trackbar(val):
    global high_V
    global low_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(track_low_V, window, low_V)


#    _______             _    _                    
#   |__   __|           | |  | |                   
#      | |_ __ __ _  ___| | _| |__   __ _ _ __ ___ 
#      | | '__/ _` |/ __| |/ / '_ \ / _` | '__/ __|
#      | | | | (_| | (__|   <| |_) | (_| | |  \__ \
#      |_|_|  \__,_|\___|_|\_\_.__/ \__,_|_|  |___/
#                                                  
#                                                  



cv2.createTrackbar(track_high_H, window , high_H, high_H, high_H_trackbar)
cv2.createTrackbar(track_high_S, window , high_S, high_S, high_S_trackbar)
cv2.createTrackbar(track_high_V, window , high_V, high_V, high_V_trackbar)

cv2.createTrackbar(track_low_H, window , low_H, high_H, low_H_trackbar)
cv2.createTrackbar(track_low_S, window , low_S, high_S, low_S_trackbar)
cv2.createTrackbar(track_low_V, window , low_V, high_V, low_V_trackbar)



#    _____                              _   _               _           _     _ _             
#   |_   _|                            | | | |             | |         | |   | (_)            
#     | |  _ __ ___   __ _  __ _  ___  | |_| |__   ___  ___| |__   ___ | | __| |_ _ __   __ _ 
#     | | | '_ ` _ \ / _` |/ _` |/ _ \ | __| '_ \ / _ \/ __| '_ \ / _ \| |/ _` | | '_ \ / _` |
#    _| |_| | | | | | (_| | (_| |  __/ | |_| | | |  __/\__ \ | | | (_) | | (_| | | | | | (_| |
#   |_____|_| |_| |_|\__,_|\__, |\___|  \__|_| |_|\___||___/_| |_|\___/|_|\__,_|_|_| |_|\__, |
#                           __/ |                                                        __/ |
#                          |___/                                                        |___/ 


if False:
    cv2.destroyAllWindows()
while True:
    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    HSV = cv2.inRange(HSV_image, (low_H, low_S, low_V), (high_H, high_S, high_V))
    cv2.imshow(window, HSV)
    
    k = cv2.waitKey(5)
    if k == ord('q'):
        break
          
cv2.destroyAllWindows()
