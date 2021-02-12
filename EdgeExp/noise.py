import cv2
import numpy as np
from matplotlib import pyplot as plt

img = 0 * np.ones((300,300),dtype=np.uint8)
cv2.circle(img, (150, 150), 80, 255, thickness=-1)

cv2.imwrite('circle.png', img)

sigma = 1
img += np.random.normal(0,sigma,img.shape).astype(np.uint8)

print(img.shape)
plt.imshow(img)
plt.show()

cv2.imwrite('noise.png', img)


