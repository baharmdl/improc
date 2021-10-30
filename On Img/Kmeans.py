import cv2
import numpy as np
import matplotlib.pyplot as plt

image= cv2.imread('butterfly.png')
image_copy= np.copy(image)
image_copy= cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

pixel_vals=image_copy.reshape(-1,3)
pixel_vals=np.float32(pixel_vals)

criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k=2
ret, labels, centers= cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers= np.unit8(centers)
segmented_data= centers[labels.flatten()]
segmented_image=segmented_data.reshape((image_copy.shape))



