import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/ellipsoid_fit')
import math

import matplotlib

import geomdl.visualization.VisMPL as VisMPL
import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline
from geomdl.fitting import approximate_surface
from geomdl.knotvector import generate
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage.segmentation import flood_fill
import nibabel as nib
import cv2
from sklearn.cluster import KMeans

import Toolbox as tb
import ellipsoid_fit as ef

center = np.array([5,3,1])
radius = np.array([3,2,1])
theta = 45
rotation = np.array([[1,0,0],
                     [0,math.cos(theta),-math.sin(theta)],
                     [0,math.sin(theta),math.cos(theta)]])

print(rotation)

psize=20

points = np.array([[radius[0]*math.cos(u)*math.cos(v), radius[1]*math.cos(v)*math.sin(u),radius[2]*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=psize) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=psize)])
for i in range(len(points)):
    points[i] = np.dot(points[i],rotation)
points += center

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ef.ellipsoid_plot(center, radius, rotation, ax=ax, plot_axes=True, cage_color='g')
ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()

print(points.shape)

center, evecs, radii = ef.ellipsoid_fit(points)

print(center)
print(evecs)
print(radii)

points = np.array([[radii[0]*math.cos(u)*math.cos(v), radii[1]*math.cos(v)*math.sin(u),radii[2]*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=psize) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=psize)])
for i in range(len(points)):
    points[i] = np.dot(points[i],evecs)
points += center

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()

surf = tb.make_ellipsoid_axis_surf(center,radius,rotation)
tb.surf_render(surf)
