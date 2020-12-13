import os
import cv2
import pickle
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
frame_nr = 0
path = r'C:/Users/Kasutaja/Desktop/Sissejuhatus andmeteadusesse/Project/test_images/'
for figure in os.listdir(path):
    str2 = "figure%d.jpg" % frame_nr
    figure = ''.join([path, str2])
    figure = cv2.imread(figure)
    cv2.imshow("figure%d" % frame_nr, figure)
    frame_nr = frame_nr+1 
    if frame_nr > 27: # how many frames to read in
        break

cv2.waitKey(0)  
cv2.destroyAllWindows()
"""

figure = mpimg.imread('./test_images/figure0.jpg')
#plt.imshow(figure); plt.title('Output'); plt.show()


#     _____                                     _ _     _             _   _                                                  _ 
#    / ____|                                   | (_)   | |           | | (_)                                                | |
#   | |     __ _ _ __ ___   ___ _ __ __ _    __| |_ ___| |_ ___  _ __| |_ _  ___  _ __    _ __ ___ _ __ ___   _____   ____ _| |
#   | |    / _` | '_ ` _ \ / _ \ '__/ _` |  / _` | / __| __/ _ \| '__| __| |/ _ \| '_ \  | '__/ _ \ '_ ` _ \ / _ \ \ / / _` | |
#   | |___| (_| | | | | | |  __/ | | (_| | | (_| | \__ \ || (_) | |  | |_| | (_) | | | | | | |  __/ | | | | | (_) \ V / (_| | |
#    \_____\__,_|_| |_| |_|\___|_|  \__,_|  \__,_|_|___/\__\___/|_|   \__|_|\___/|_| |_| |_|  \___|_| |_| |_|\___/ \_/ \__,_|_|
#                                                                                                                              
#                                                                                                                              

calibration_pickle = pickle.load( open( "./calibration_matrix.p", "rb" ) ) #manufacturer calibration matrix for lens distortion removal
m = calibration_pickle["m"]
d = calibration_pickle["dist"]

img_undist = cv2.undistort(figure, m, dist, None, m)
#plt.figure(1); plt.imshow(img_undist); plt.title('Undistorted image', fontsize=30); # plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,15))
f.tight_layout()
ax1.set_title('Original Image', fontsize=30)
ax1.imshow(figure)
ax2.set_title('Undistorted image', fontsize=30)
ax2.imshow(img_undist)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()


#    _    _  _______      __            _                                     
#   | |  | |/ ____\ \    / /           | |                                    
#   | |__| | (___  \ \  / /    ___ ___ | | ___  _ __ ___ _ __   __ _  ___ ___ 
#   |  __  |\___ \  \ \/ /    / __/ _ \| |/ _ \| '__/ __| '_ \ / _` |/ __/ _ \
#   | |  | |____) |  \  /    | (_| (_) | | (_) | |  \__ \ |_) | (_| | (_|  __/
#   |_|  |_|_____/    \/      \___\___/|_|\___/|_|  |___/ .__/ \__,_|\___\___|
#                                                       | |                   
#                                                       |_|                   



blur = cv2.bilateralFilter(img_undist,9,100,100) # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed:~:text=bilateralFilter()
#plt.figure(2); plt.imshow(blur); plt.title('Filtered image (bilateral filter) nr 2', fontsize=30); plt.show()
hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
plt.figure(4); plt.imshow(hsv); plt.title('HSV image', fontsize=30)




#    _    _  _______      __                _   _                  _       _   
#   | |  | |/ ____\ \    / /               | | | |                | |     | |  
#   | |__| | (___  \ \  / /   ___  ___ __ _| |_| |_ ___ _ __ _ __ | | ___ | |_ 
#   |  __  |\___ \  \ \/ /   / __|/ __/ _` | __| __/ _ \ '__| '_ \| |/ _ \| __|
#   | |  | |____) |  \  /    \__ \ (_| (_| | |_| ||  __/ |  | |_) | | (_) | |_ 
#   |_|  |_|_____/    \/     |___/\___\__,_|\__|\__\___|_|  | .__/|_|\___/ \__|
#                                                           | |                
#                                                           |_|                


"""
u_white = np.array([36, 255, 255])
l_white = np.array([15,0,0])

HSV_pixels = hsv.reshape((np.shape(hsv)[0]*np.shape(hsv)[1], 3))
normalize = colors.Normalize(vmin=-1.,vmax=1.)
normalize.autoscale(HSV_pixels)
normalized_colors = norm(HSV_pixels).tolist()

h, s, v = cv2.split(hsv)
fig = plt.figure(3)
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=normalized_colors, marker="x")
axis.set_xlabel("H")
axis.set_ylabel("S")
axis.set_zlabel("V")
plt.title('HSV colorspace for one of the input video frames', fontsize=20); plt.show()
"""


#    _    _  _______      __  _   _                   _           _     _                       _ _   
#   | |  | |/ ____\ \    / / | | | |                 | |         | |   | |                     | | |  
#   | |__| | (___  \ \  / /  | |_| |__  _ __ ___  ___| |__   ___ | | __| |  _ __ ___  ___ _   _| | |_ 
#   |  __  |\___ \  \ \/ /   | __| '_ \| '__/ _ \/ __| '_ \ / _ \| |/ _` | | '__/ _ \/ __| | | | | __|
#   | |  | |____) |  \  /    | |_| | | | | |  __/\__ \ | | | (_) | | (_| | | | |  __/\__ \ |_| | | |_ 
#   |_|  |_|_____/    \/      \__|_| |_|_|  \___||___/_| |_|\___/|_|\__,_| |_|  \___||___/\__,_|_|\__|
#                                                                                                     
#                                                                                                     


mask = cv2.inRange(hsv, (0,0,0), (255, 255, 163))

# Bitwise AND mask
#mask = cv2.bitwise_or(mask1, mask2)
res = cv2.bitwise_and(blur,blur, mask= mask)
#cv.imshow('frame',frame)
#cv.imshow('mask',mask)

#plt.figure(4); plt.imshow(mask); plt.title('Mask', fontsize=30)   
plt.figure(5); plt.imshow(res); plt.title('Thresholded with found HSV parameters', fontsize=30); plt.show()







