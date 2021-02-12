import sys; sys.path.append('/home/kitamiya/Documents/python/snake3D/src/Toolbox')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import Toolbox as tb
import cv2

res = 100
center = res/2
radius_true = 25

sphere = tb.make_sphere_voxel(res,center,radius_true)

sphere_center = np.array([center,center,center]) 
clip_center = np.array([59.44503706,46.78571999,78.2933893])
direction = np.array([0.31245829,-0.11096634,0.94342795])

rotated = tb.clip_rect(sphere,clip_center,(19,6,6),direction)
# new_rotated = tb.new_clip_rect(sphere,clip_center,(20,6,6),direction)
print(rotated.shape)

# print(np.linalg.norm(sphere_center - clip_center))

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.voxels(rectangular)
ax.voxels(rotated)

plt.show()