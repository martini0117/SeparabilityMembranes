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

import Toolbox as tb

rect_size = np.array([20,6,6])

# img = tb.load_test_medical_image()
# tb.show_ct_image(img)

margin = rect_size[0]
inside = tb.load_test_medical_image()
# tb.show_ct_image(inside)

img_shape = tuple([s + 2*margin for s in inside.shape])

img = np.zeros(img_shape)
img += np.min(inside)
img[margin:img_shape[0]-margin,margin:img_shape[1]-margin,margin:img_shape[2]-margin] += inside - np.min(inside)

tb.show_ct_image(img)