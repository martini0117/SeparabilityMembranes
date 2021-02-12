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

from sklearn.cluster import DBSCAN

case = 210

data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'

case_str = tb.get_full_case_id(case)

nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
gt = nii0.get_data().astype(np.uint8)

# tb.show_ct_image(gt)

nii0=nib.load(data_path + case_str + '/prediction.nii.gz')
seg = nii0.get_data().astype(np.uint8)

tb.show_ct_image(seg)


nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
img = nii0.get_data()

# img = tb.threshold_image_seg(img,seg)
# t_min = -80
# t_max = 200
# img = tb.threshold_image_minmax(img,t_min,t_max)

# tb.show_ct_image(img)


tb.show_ct_image(tb.draw_segmentation(img,gt,mark_val=2))

rect_size = np.array([15,3,3])
margin = rect_size[0]
img = tb.add_margin(img, margin)
seg = tb.add_margin(seg, margin)


# tb.show_ct_image(img)
# tb.show_ct_image(seg)

seg = np.where(seg==2,1,0).astype(np.uint8)

tb.show_ct_image(tb.draw_segmentation(img,seg))

contour_pts = tb.get_full_pts(seg)


pred = np.array(DBSCAN(eps=1).fit_predict(contour_pts))
num_clu = np.max(pred) + 1

selected_clu = np.argmax(np.array([np.sum(pred==i) for i in range(num_clu)]))

not_selected = contour_pts[pred != selected_clu,:]

removed_seg = seg.copy()

# print(not_selected.shape[0])

for i in range(not_selected.shape[0]):
    removed_seg[not_selected[i,0],not_selected[i,1],not_selected[i,2]] = 0

contour_pts = tb.get_contour_pts(removed_seg)

tb.show_ct_image(removed_seg)



# tb.show_ct_image(removed_seg)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i in range(-1,num_clu):
#     ax.scatter(contour_pts[pred==i,0], contour_pts[pred==i,1], contour_pts[pred==i,2])
# plt.show()

# center, radius = tb.sphere_fit(contour_pts[pred==c_no,0],contour_pts[pred==c_no,1],contour_pts[pred==c_no,2])
# radius -= 5

center, evecs, radii = ef.ellipsoid_fit(contour_pts)
print(center)
print(radii)

print(4/3*math.pi*radii[0]*radii[1]*radii[2])
print(np.sum(removed_seg))

surf = tb.make_ellipsoid_axis_surf(center,radii,evecs)
color_img = tb.draw_contour(seg,surf)
tb.show_ct_image(color_img)
# color_img = tb.draw_contour(img,surf)
# tb.show_ct_image(color_img)


# surf = tb.cul_contour(seg,surf,rect_size,div=20,N_limit=5,c=0.95,ctrlpts_size=6,dif_limit=0.001,dif_abs=False,w=0.95)

# tb.surf_render(surf)

# color_img = tb.draw_contour(seg,surf)
# tb.show_ct_image(color_img)

# surf = tb.cul_contour(img,surf,rect_size,div=20,N_limit=30,c=0.95,ctrlpts_size=8,dif_limit=0.001,dif_abs=False,w=0.95,ctrlpts_increasing=True,increasing_limit=0.05)
surf = tb.cul_contour_seg(img,seg,surf,rect_size,div=20,N_limit=30,c=0.8,ctrlpts_size=8,dif_limit=0.001,dif_abs=False,w=0.95,ctrlpts_increasing=True,increasing_limit=0.05,weight=0.3)

# tb.surf_render(surf)

# color_img = tb.draw_contour(img,surf)
# tb.show_ct_image(color_img)

img_mask =tb.get_image_mask(img,surf)
con_seg = tb.fill_contour(img_mask)
# tb.show_ct_image(img_mask)
# tb.show_ct_image(con_seg)

seg = seg[margin:con_seg.shape[0]-margin,margin:con_seg.shape[1]-margin,margin:con_seg.shape[2]-margin]
seg_original = np.where(seg==1,1,seg_original)

save_con_seg = con_seg[margin:con_seg.shape[0]-margin,margin:con_seg.shape[1]-margin,margin:con_seg.shape[2]-margin]
save_con_seg = np.where(save_con_seg==255, 2, seg_original)

tb.show_ct_image(tb.draw_segmentation(img,con_seg,mark_val=255))
