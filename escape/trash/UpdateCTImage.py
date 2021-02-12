import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import math

import geomdl.visualization.VisMPL as VisMPL
import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline
from geomdl.fitting import approximate_surface
from geomdl.knotvector import generate
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import Toolbox as tb

img = tb.load_test_medical_image()
tb.show_ct_image(img)

center = np.array([140,300,340])
radius = 40

surf = tb.make_sphere_surf(center,radius)

rect_size = np.array([20,6,6])

surf = tb.cul_contour(img,surf,rect_size)


surf.delta = 0.05
surf.vis = VisMPL.VisSurfWireframe()
surf.render()