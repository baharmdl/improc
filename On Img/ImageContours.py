import cv2
import numpy as np
import matplotlib.pyplot as plt

image= cv2.imread('hand.png')
image_copy= np.copy(image)

gray= cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

#1
ret, binary= cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
plt.imshow(binary, cmap='gray')
plt.show()

#2
# binary= cv2.inRange(gray, 0, 225)
# plt.imshow(binary, cmap='gray')
# plt.show()

#3
# binary= cv2.Canny(gray, 80, 200)
# plt.imshow(binary)
# plt.show()

contours, hierachy= cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_copy2=np.copy(image_copy)
all_contours= cv2.drawContours(image_copy2, contours, -1, (250, 100,150), 2)
plt.imshow(all_contours)
plt.show()



