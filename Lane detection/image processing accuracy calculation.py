import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./test_images/figure0.jpg')
image = cv2.GaussianBlur(image,(13,13),0)
#image = cv2.bilateralFilter(image,5,100,100)
#image = cv2.medianBlur(image,7)

limits = [([0, 0, 0], [255, 255, 163])] # Line colored pixels in HSV colorspace

for low_l, high_l in limits:
    
    # array of low and high limits
    low_l= np.array(low_l)
    high_l = np.array(high_l)
    # Bitwise and mask
    mask = cv2.inRange(image, low_l, high_l)
    result = cv2.bitwise_and(image, image, mask = mask)
    
plt.figure(3); plt.imshow(mask); plt.title('Mask', fontsize=30)
plt.figure(4); plt.imshow(result); plt.title('Result image', fontsize=30)  ; plt.show()

total_pixels = result.size
pixels = total_pixels - np.count_nonzero(result)
percent = round(pixels * 100 / total_pixels, 1)

print("Line colored pixels: " + str(pixels))
print("Total nr of pixels: " + str(total_pixels))
print("Percentage of line colored pixels: " + str(percent) + "%")

