# %%
import sys; sys.path.append('/home/kitamiya/Documents/python/snake3D/src')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import Toolbox as tb

#%%
# MakeSphereVoxel
res = 50
center = res/2
radius = center*3/4

sphere = tb.make_sphere_voxel(res,center,radius)

print(sphere.dtype)

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(sphere)

plt.show()

#%%
# make_rectangular_voxel
res = 50
wdh = [10,30,20]
position = [10,10,10]

rectangular = tb.make_rectangular_voxel(res,wdh,position)


# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(rectangular)

plt.show()

#%%
# clip_rect
res = 50
wdh = [1,1,25]
position = [25,25,25]

rectangular = tb.make_rectangular_voxel(res,wdh,position)

rotated = tb.clip_rect(rectangular,(25,25,25),(10,10,10),np.array([-1,0,1]))

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.voxels(rectangular)
ax.voxels(rotated)

plt.show()

#%%
# cul_normal_vector
from geomdl import BSpline
from geomdl.knotvector import generate
from geomdl.fitting import approximate_surface
import geomdl.visualization.VisMPL as VisMPL
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = np.array([[math.cos(u)*math.cos(v), math.cos(v)*math.sin(u),math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=30) 
                    for v in np.linspace(-math.pi/2,math.pi/2,num=30)])

surf = approximate_surface(points.tolist(),30,30,3,3,ctrlpts_size_u=10,ctrlpts_size_v=10)

div = 100
evalpts = np.array([surf.evaluate_list([(u,v) for v in np.linspace(0,1,div)]) for u in np.linspace(0,1,div)])

u = 8
v = 8

center = evalpts[u,v,:]
neighborhood = np.array([evalpts[u-1,v,:],evalpts[u,v-1,:],evalpts[u+1,v,:],evalpts[u,v+1,:]])

normal = tb.cul_normal_vector(center, neighborhood)
print(normal)

#%%
#cul_separability
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

w = 20
h = 6
d = 6

N = w*h*d

clipped_image = np.zeros((w,h,d))
clipped_image[:int(w/2),:,:] = np.random.randn(int(w/2),h,d)
clipped_image[int(w/2):,:,:] = np.random.randn(int(w/2),h,d) + 1

print(tb.cul_separability(clipped_image))
# print(variance_boundary/variance_all)