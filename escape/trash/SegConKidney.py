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

case = 'case_00073'

nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/' + case + '/imaging.nii.gz')
img = nii0.get_data()

nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/' + case + '/kidney.nii.gz')
seg = nii0.get_data().astype(np.uint8)

tb.show_ct_image(img)
tb.show_ct_image(seg)

# rect_size = np.array([20,6,6])
# img = tb.add_margin(img, rect_size[0])

# tb.show_ct_image(tb.draw_segmentation(img,seg))

img_contour = np.zeros(seg.shape,dtype=np.uint8)
contour_pts = []

for i in range(seg.shape[0]):
    contour,_ = cv2.findContours(seg[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contour:
        for pts in c:
            contour_pts.append([i,pts[0,1],pts[0,0]])

    img_contour[i,:,:] = cv2.polylines(img_contour[i,:,:],contour,True,255)
# print(len(contour_pts))
# tb.show_ct_image(img_contour)

contour_pts = np.array(contour_pts)
pred = KMeans(n_clusters=2,random_state=173).fit_predict(contour_pts)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i in range(2):
#     ax.scatter(contour_pts[pred==i,0], contour_pts[pred==i,1], contour_pts[pred==i,2])
# plt.show()


c_no = 0

# center, radius = tb.sphere_fit(contour_pts[pred==c_no,0],contour_pts[pred==c_no,1],contour_pts[pred==c_no,2])
# radius -= 5

center, evecs, radii = ef.ellipsoid_fit(contour_pts[pred==c_no,:])
print(center)

surf = tb.make_ellipsoid_axis_surf(center,radii,evecs)
color_img = tb.draw_contour(seg,surf)
tb.show_ct_image(color_img)

rect_size = np.array([20,6,6])

surf = tb.cul_contour(seg,surf,rect_size,div=20,N_limit=5,c=0.95,ctrlpts_size=8,dif_limit=0.001,w=0.95)

tb.surf_render(surf)

color_img = tb.draw_contour(seg,surf)
tb.show_ct_image(color_img)

surf = tb.cul_contour(seg,surf,rect_size,div=20,N_limit=30,c=0.95,ctrlpts_size=8,dif_limit=0.001,dif_abs=False,w=0.95)

tb.surf_render(surf)

color_img = tb.draw_contour(img,surf)
tb.show_ct_image(color_img)
