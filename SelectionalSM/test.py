import numpy as np

indices = np.array([[-1, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

img_shape = [10,10,10]
inside = [np.logical_and(i <= indices[:,i],indices[:,i] < img_shape[i]) for i in range(3)]
print(inside)
inside = np.all(inside, axis=0)

indices = indices[inside]
print(indices)