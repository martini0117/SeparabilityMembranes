import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('noise.png')
edges = cv2.Canny(img,100,200)
cv2.imwrite('noise_edge.png', edges)

img = cv2.imread('circle.png')
edges = cv2.Canny(img,100,200)
cv2.imwrite('circle_edge.png', edges)

img = cv2.imread('blur.png')
edges = cv2.Canny(img,100,200)
cv2.imwrite('blur_edge.png', edges)

plt.imshow(edges)
# plt.show()