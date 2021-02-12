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
from sklearn.cluster import DBSCAN

import Toolbox as tb
import ellipsoid_fit as ef

from sklearn.neighbors import BallTree


# cases = range(150,190)
cases = [155]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/KidneyContour/'

window_center = 0
window_width = 300

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)

    # データ読み込み
    nii0=nib.load(data_path + case_str + '/kidney_contour.nii.gz')
    pre = nii0.get_data()
    pre_con = tb.get_full_pts(pre)

    nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
    gt = nii0.get_data()
    gt_con = tb.get_contour_pts(gt)

    tree = BallTree(gt_con)              
    _, ind = tree.query(pre_con, k=1)
    ind = np.reshape(ind,(ind.shape[0]))

    dist_sum = 0
    for i in range(pre_con.shape[0]):
        dist_sum += np.linalg.norm(pre_con[i,:] - gt_con[ind[i],:])
    