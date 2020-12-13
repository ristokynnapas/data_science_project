import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


start_time = time.time() # time measurment

img = cv2.imread('./test_images/figure0.jpg') # insert image



#     _____          __  __ ______ _____              _____ _____  _____ _______ ____  _____ _______ _____ ____  _   _   _____  ______ __  __  ______      __     _      
#    / ____|   /\   |  \/  |  ____|  __ \     /\     |  __ \_   _|/ ____|__   __/ __ \|  __ \__   __|_   _/ __ \| \ | | |  __ \|  ____|  \/  |/ __ \ \    / /\   | |     
#   | |       /  \  | \  / | |__  | |__) |   /  \    | |  | || | | (___    | | | |  | | |__) | | |    | || |  | |  \| | | |__) | |__  | \  / | |  | \ \  / /  \  | |     
#   | |      / /\ \ | |\/| |  __| |  _  /   / /\ \   | |  | || |  \___ \   | | | |  | |  _  /  | |    | || |  | | . ` | |  _  /|  __| | |\/| | |  | |\ \/ / /\ \ | |     
#   | |____ / ____ \| |  | | |____| | \ \  / ____ \  | |__| || |_ ____) |  | | | |__| | | \ \  | |   _| || |__| | |\  | | | \ \| |____| |  | | |__| | \  / ____ \| |____ 
#    \_____/_/    \_\_|  |_|______|_|  \_\/_/    \_\ |_____/_____|_____/   |_|  \____/|_|  \_\ |_|  |_____\____/|_| \_| |_|  \_\______|_|  |_|\____/   \/_/    \_\______|
#                                                                                                                                                                        
#                                                                                                                                                                        

"""
import pickle
calibration_pickle = pickle.load( open( "./calibration_matrix.p", "rb" ) )
m = calibration_pickle["mtx"]
d = calibration_pickle["dist"]

img_undist = cv2.undistort(img, m, d, None, m)

#    _______ _                   _           _     _ _                     _            _ _            
#   |__   __| |                 | |         | |   | (_)                   (_)          | (_)           
#      | |  | |__  _ __ ___  ___| |__   ___ | | __| |_ _ __   __ _   _ __  _ _ __   ___| |_ _ __   ___ 
#      | |  | '_ \| '__/ _ \/ __| '_ \ / _ \| |/ _` | | '_ \ / _` | | '_ \| | '_ \ / _ \ | | '_ \ / _ \
#      | |  | | | | | |  __/\__ \ | | | (_) | | (_| | | | | | (_| | | |_) | | |_) |  __/ | | | | |  __/
#      |_|  |_| |_|_|  \___||___/_| |_|\___/|_|\__,_|_|_| |_|\__, | | .__/|_| .__/ \___|_|_|_| |_|\___|
#                                                             __/ | | |     | |                        
#                                                            |___/  |_|     |_|                        


blur = cv2.GaussianBlur(img_undist,(9,9),0)
#blur = cv2.bilateralFilter(img,5,100,100)
#blur = cv2.medianBlur(img,7)

hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0,0,0), (255, 255, 163)) ## ranges where found by thresholding
res = cv2.bitwise_and(blur,blur, mask= mask)
"""

blur = cv2.GaussianBlur(img,(5,5),0)
#blur = cv2.bilateralFilter(img,5,100,100)
#blur = cv2.medianBlur(img,7)
edges = cv2.Canny(blur,50,200) #https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de

plt.figure()
plt.imshow(edges)
#plt.show()

"""
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Canny edge detection with median blur, kernel size 7x7'), plt.xticks([]), plt.yticks([])
plt.show()
"""

############ Laplacian
"""
laplacian = cv2.Laplacian(blur,5,cv2.CV_64F)
#laplacian = cv2.Laplacian(img,cv2.CV_8UC1)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian edge detection with median blur, kernel size 5x5'), plt.xticks([]), plt.yticks([]), plt.show()
"""

############ Sobel
            
"""            
# Sobel gradient - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
sobelx64f = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)            
            
plt.figure()
plt.imshow(sobel_8u)
plt.show()

"""


#    _____                                             _              
#   |_   _|                                           (_)             
#     | |  _ __ ___   __ _  __ _  ___   _ __ ___  __ _ _  ___  _ __   
#     | | | '_ ` _ \ / _` |/ _` |/ _ \ | '__/ _ \/ _` | |/ _ \| '_ \  
#    _| |_| | | | | | (_| | (_| |  __/ | | |  __/ (_| | | (_) | | | | 
#   |_____|_| |_| |_|\__,_|\__, |\___| |_|  \___|\__, |_|\___/|_| |_| 
#                           __/ |                 __/ |               
#                          |___/                 |___/                


def ROI(img, points):
    
    mask = np.zeros_like(img)   
    color = 255
    
    cv2.fillPoly(mask, points, color)
    
    result_image = cv2.bitwise_and(img, mask) 
    return result_image # image that has non_zero values will be returned
 
 
height, width, channels = img.shape
print("The image has the following shape:", height, width, channels)

# mask points coordinates - these define the shape

#high_left = [0, 0]
#low_left = [0, 100]

#high_right = [640, 0]
#low_right = [640, 100]

high_left = [0, 75]
low_left = [0, 368]

high_right = [500, 100]
low_right = [640, 368]

coordinates = np.array([[low_left, high_left, high_right, 
low_right]], dtype=np.int32)

cropped_image = ROI(edges, coordinates)

plt.subplot(121),plt.imshow(edges)
plt.title('Edge detection image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cropped_image)
plt.title('Canny edge detection with region of interest selecting'), plt.xticks([]), plt.yticks([])
plt.show()


#    _      _                  _      _            _   _             
#   | |    (_)                | |    | |          | | (_)            
#   | |     _ _ __   ___    __| | ___| |_ ___  ___| |_ _  ___  _ __  
#   | |    | | '_ \ / _ \  / _` |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
#   | |____| | | | |  __/ | (_| |  __/ ||  __/ (__| |_| | (_) | | | |
#   |______|_|_| |_|\___|  \__,_|\___|\__\___|\___|\__|_|\___/|_| |_|
#                                                                    
#                                                                    


def hough_transform(img):
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
              minLineLength=min_line_len, maxLineGap=max_line_gap) # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[0, 128, 0], thickness=3): ## unfortunately lost it - but used a good stackoverflow page for that
    
    if lines is None:
        return
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
  
#hough parameters
rho = 1
theta = np.pi/180
threshold = 10
min_line_len = 10
max_line_gap = 10

hough_t = hough_transform(edges)
plt.figure()
plt.imshow(hough_t)
#plt.show()

"""
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(hough_t)
plt.title('Hough edge annotation - best result. MaxLineGap 5'), plt.xticks([]), plt.yticks([])
#plt.show()
"""


def color_img(img, initial_img, α=0.7, β=1, λ=0):
    
    return cv2.addWeighted(initial_img, α, img, β, λ)
  
"""
colored_image = color_img(hough_t, img)
plt.figure()
plt.imshow(colored_image)
plt.show()
"""

#   __      ___     _                                      _        _   _             
#   \ \    / (_)   | |                                    | |      | | (_)            
#    \ \  / / _  __| | ___  ___     __ _ _ __  _ __   ___ | |_ __ _| |_ _  ___  _ __  
#     \ \/ / | |/ _` |/ _ \/ _ \   / _` | '_ \| '_ \ / _ \| __/ _` | __| |/ _ \| '_ \ 
#      \  /  | | (_| |  __/ (_) | | (_| | | | | | | | (_) | || (_| | |_| | (_) | | | |
#       \/   |_|\__,_|\___|\___/   \__,_|_| |_|_| |_|\___/ \__\__,_|\__|_|\___/|_| |_|
#                                                                                     
#                                                                                     


# Video to annotate
cap = cv2.VideoCapture('joon.avi')
 
if (cap.isOpened() == False): 
    print("Error opening video stream or file")
    
########## Writing video   
"""    
size = (int(cap.get(3)) , int(cap.get(4))) 
result = cv2.VideoWriter('proof_of_concept4.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                        20, size)
"""

# Read until video is completed
print("Video started")
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        blur = cv2.GaussianBlur(frame,(5,5),0)
        #blur = cv2.bilateralFilter(blur,5,100,100)
        #blur = cv2.medianBlur(blur,7)
        
        #laplacian = cv2.Laplacian(blur,5,cv2.CV_64F)
        edges = cv2.Canny(blur,50,200)
        
        cropped_image = ROI(edges, coordinates)
        hough_t = hough_transform(cropped_image)
        colored_image = color_img(hough_t, frame)
        
        cv2.imshow('Frame', colored_image)
        #result.write(colored_image) # writing video
        time.sleep(0.01)
    if ret == False:
        break
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
print("%d seconds" % (time.time() - start_time))   
print("Video ended")
cap.release()
cv2.destroyAllWindows()
