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

import Toolbox as tb


# 腎臓
inside = tb.load_test_medical_image()

rect_size = np.array([20,6,6])
center = np.array([135,300,330])
radius = np.array([35,35,35])

margin = rect_size[0]
center += margin

img = tb.add_margin(inside, margin)
img = tb.window_function(img,0,300)
# tb.show_ct_image(img)

bsurf = tb.make_ellipsoid_surf(center,radius)
# surf.delta = 0.001
# init_points = np.array(surf.evalpts)
# pnum = int(np.sqrt(init_points.shape[0]))
# init_points = np.reshape(init_points,(pnum,pnum,3))
# print(init_points.shape)
# tb.show_ct_image(tb.draw_contour(img,surf))

mask_sum = np.zeros(img.shape,dtype=np.uint8)

for gu in range(2):
    for gv in range(2):
        wide = 15
        psize = 100
        qsize = 4
        # patch_center = init_points[gu,gv,:] + margin
        # patch_u = init_points[gu+1,gv,:] - init_points[gu,gv,:]
        # patch_u = patch_u / np.linalg.norm(patch_u) 
        # patch_v = init_points[gu,gv+1,:] - init_points[gu,gv,:]
        # pathc_v = patch_v / np.linalg.norm(patch_v)

        # points = np.array([patch_center + u * patch_u + v * pathc_v 
        #                 for u in np.linspace(-wide,wide,num=psize) 
        #                 for v in np.linspace(-wide,wide,num=psize)])
        bsurf.delta = 0.05
        bsurf.evaluate(start_u=0.5*gu,stop_u=0.5*(gu+1),start_v=0.5*gv,stop_v=0.5*(gv+1))
        points = np.array(bsurf.evalpts)
        psize = int(np.sqrt(points.shape[0]))
        # points = np.reshape(points,(psize,psize,3))
        surf = approximate_surface(points.tolist(),psize,psize,3,3,ctrlpts_size_u=qsize,ctrlpts_size_v=qsize)
        # tb.show_image_collection(tb.draw_contour(img,surf).astype(np.uint8))

        surf = tb.cul_contour(img,surf,rect_size,div=20,N_limit=10,c=0.95,ctrlpts_size=qsize,dif_limit=0.001,w=0.95)
        mask = tb.get_image_mask(img, surf)
        mask_sum = np.where(np.logical_or(mask_sum == 255, mask == 255),255,0)
        

        # tb.surf_render(surf)

        # color_img = tb.draw_contour(img, surf).astype(np.uint8)

        # tb.show_ct_image(color_img)

# tb.show_image_collection(mask_sum)
final = tb.draw_segmentation_u(img, mask_sum, mark_val=255).astype(np.uint8)
tb.show_image_collection(final)
