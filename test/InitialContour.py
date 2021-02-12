import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
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

nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/case_00031/kidney.nii.gz')
img = nii0.get_data().astype(np.uint8)

rect_size = np.array([20,6,6])
img = tb.add_margin(img, rect_size[0])

tb.show_ct_image(img)

img_contour = np.zeros(img.shape,dtype=np.uint8)
contour_pts = []

for i in range(img.shape[0]):
    contour,_ = cv2.findContours(img[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # for c in contour:
    #     for pts in c:
    #         print(pts.shape)
    #         contour_pts.append([i,pts[0,0],pts[0,1]])

    for c in contour:
        for pts in c:
            # print(pts.shape)
            contour_pts.append([i,pts[0,0],pts[0,1]])

    
    # img_contour[i,:,:] = cv2.polylines(img_contour[i,:,:],contour,True,255)
print(len(contour_pts))
# tb.show_ct_image(img_contour)

contour_pts = np.array(contour_pts)
pred = KMeans(n_clusters=2).fit_predict(contour_pts)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(contour_pts[pred==1,0], contour_pts[pred==1,1], contour_pts[pred==1,2])
plt.show()

center, radius = tb.sphere_fit(contour_pts[pred==1,0],contour_pts[pred==1,1],contour_pts[pred==1,2])
radius -= 5
print(center)
print(radius)
# center = np.array([55,215,120])
# radius = 25

print(center, radius)

surf = tb.make_sphere_surf(center,radius)
surf.delta = 0.05
surf.vis = VisMPL.VisSurfWireframe()
surf.render()

surf = tb.cul_contour(img,surf,rect_size,div=20,N_limit=5,c=0.95,ctrlpts_size=8,dif_limit=0.001,w=0.95)

surf.delta = 0.05
surf.vis = VisMPL.VisSurfWireframe()
surf.render()
