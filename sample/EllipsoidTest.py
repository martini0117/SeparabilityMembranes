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

center = np.array([100,100,100])
radius = np.array([10,20,30])

surf = tb.make_ellipsoid_surf(center, radius)

surf.delta = 0.05
surf.vis = VisMPL.VisSurfWireframe()
surf.render()