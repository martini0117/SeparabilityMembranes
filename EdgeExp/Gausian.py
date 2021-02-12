import cv2
import numpy as np
from matplotlib import pyplot as plt

img = 0 * np.ones((300,300),dtype=np.uint8)
cv2.circle(img, (150, 150), 80, 255, thickness=-1)

cv2.imwrite('circle.png', img)

for i in range(100):
    img = cv2.blur(img,(5,5),0)

print(img.shape)
plt.imshow(img)
plt.show()

cv2.imwrite('blur.png', img)