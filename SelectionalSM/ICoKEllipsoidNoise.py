import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/ellipsoid_fit')
import math

import matplotlib

from geomdl.exchange import export_json
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
from sklearn.cluster import DBSCAN

import os

import Toolbox as tb
import ellipsoid_fit as ef

np.random.seed(seed=173)

#160,181,189
# cases = range(150,190)
# cases = range(150,160)
cases = [150]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'

#ノイズを加えた初期輪郭を複数作成する

window_center = 0
window_width = 300

human = '_h'

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)

    # 初期輪郭取得
    center = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_center' + human + '.npy')
    radii = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_radius' + human + '.npy')
    evecs = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_rotation' + human + '.npy')
    points = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_points' + human + '.npy')

    original_surf = tb.make_ellipsoid_axis_surf(center,radii,evecs)
    # tb.surf_render(surf)

    output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/contours/kidney_EN/' + case_str + '/'
    os.makedirs(output_path,exist_ok=True)
    num_of_surfs = 10
    for s in range(num_of_surfs):
        if s == 0:
            save_surf = original_surf
        else:
            oscnp = np.array(original_surf.ctrlpts)
            oscnp += np.random.standard_normal(oscnp.shape)
            
            moved_surf = original_surf
            moved_surf.set_ctrlpts(oscnp.tolist(), original_surf.ctrlpts_size_u, original_surf.ctrlpts_size_v)
            # tb.surf_render(moved_surf)
            print(moved_surf.ctrlpts)
            save_surf = moved_surf
        
        export_json(save_surf,output_path + str(s) + '.json')
        

            
            
