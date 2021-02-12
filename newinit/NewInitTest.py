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
from geomdl.fitting import interpolate_curve
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from matplotlib import cm

import Toolbox as tb

points = [[121,335,329],[126,337,333],[141,335,340],[159,327,355],[181,323,358],[194,315,354],[198,315,355]]
radius = [2,15,30,30,25,15,2]
pn = 20
qn = 8

surf = tb.manual_init(points, radius, pn,qn)

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

# surf = tb.make_ellipsoid_surf(center,radius)
tb.show_ct_image(tb.draw_contour(img,surf))

surf = tb.cul_contour(img,surf,rect_size,div=20,N_limit=30,c=0.95,ctrlpts_size=8,dif_limit=0.001,w=0.95)

surf.vis = VisMPL.VisSurface()
surf.render(colormap=cm.cool)

color_img = tb.draw_contour(img, surf)

tb.show_ct_image(color_img)


