import cv2
import os


#     _____             _               ______                              
#    / ____|           (_)             |  ____|                             
#   | (___   __ ___   ___ _ __   __ _  | |__ _ __ __ _ _ __ ___   ___  ___  
#    \___ \ / _` \ \ / / | '_ \ / _` | |  __| '__/ _` | '_ ` _ \ / _ \/ __| 
#    ____) | (_| |\ V /| | | | | (_| | | |  | | | (_| | | | | | |  __/\__ \ 
#   |_____/ \__,_| \_/ |_|_| |_|\__, | |_|  |_|  \__,_|_| |_| |_|\___||___/ 
#                                __/ |                                      
#                               |___/                                       


cap = cv2.VideoCapture('joon.avi')

if (cap.isOpened() == False): 
    print("Error opening video stream or file")
print("Video started")
frame_nr = 0
path = 'C:/Users/Kasutaja/Desktop/Sissejuhatus andmeteadusesse/Project/test_images' ## Path - remember to change!

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        cv2.imshow('Frame', frame)
        cv2.imwrite(os.path.join(path, "figure%d.jpg" % frame_nr), frame)
        frame_nr += 1
    if ret == False:
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print("Video ended")
cap.release()
cv2.destroyAllWindows()