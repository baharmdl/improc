import cv2
import numpy as np
import matplotlib.pyplot as plt

image= cv2.imread('chess.png')
image_copy= np.copy(image)
gray= cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)


gray= np.float32(gray)
dst= cv2.cornerHarris(gray, 2, 3, 0.04)
# plt.imshow(dst)
# plt.show()
# cv2.imshow('yoohooo',dst)
# cv2.waitKey(0)
dst_dilate= cv2.dilate(dst, None)
# plt.imshow(dst_dilate)
# plt.show()
# cv2.imshow('yoohooo',dst)
# cv2.waitKey(0)


gray_copy= np.copy(gray)
thresh= 0.001* dst_dilate.max()
for i in range (0, gray_copy.shape[0]):
    for j in range (0, gray_copy.shape[1]):
        if (dst_dilate[i, j]> thresh):
            image_copy[i, j]=[255, 100, 0]

plt.imshow(image_copy)
plt.show()



