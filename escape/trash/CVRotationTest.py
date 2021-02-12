import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

img = np.zeros((300,400))

img[100:120,300:320] = 1
img[100,200] = 1

fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')

plt.show()

M = cv2.getRotationMatrix2D((200,100),30,1)
img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))

fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')

plt.show()
