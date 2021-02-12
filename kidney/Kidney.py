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

surf = tb.make_ellipsoid_surf(center,radius)
tb.show_ct_image(tb.draw_contour(img,surf))

surf = tb.cul_contour(img,surf,rect_size,div=20,N_limit=30,c=0.95,ctrlpts_size=8,dif_limit=0.001,w=0.95)

tb.surf_render(surf)

color_img = tb.draw_contour(img, surf)

tb.show_ct_image(color_img)


