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
cases = range(150,190)
# cases = range(150,160)
# cases = [150]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
img_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'

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

    #初期輪郭をサーフェイスで
    original_surf = tb.make_ellipsoid_axis_surf(center,radii,evecs)
    # tb.surf_render(surf)

    #ノイズを乗せた初期輪郭を作成
    save_surfs = tb.make_surfaces_by_snd(original_surf)

    #ボリュームデータの読み込み
    nii0 = nib.load(img_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()
    img = tb.window_function(img,window_center,window_width)

    rect_size = np.array([30,6,6])

    #すべての初期輪郭に対して分離度膜を適用
    result_surfs = []

    for init in save_surfs:
        final = tb.separability_membrane(img, init, rect_size,debug=False)
        result_surfs.append(final)

    #最終結果の分離度を計算
    rs_separabilities = []
    for rs in result_surfs:
        bb = tb.bb_from_surface(rs, img)

        separability = tb.separability_via_bb(rs, img, bb)
        rs_separabilities.append(separability)

    #分離度が一番高かった結果を選ぶ
    final_result = result_surfs[np.argmax(rs_separabilities)]

    #精度を計算
    nii0=nib.load(img_path + case_str + '/segmentation.nii.gz')
    gt = nii0.get_data().astype(np.uint8)
    gt = np.where(gt==2,0,gt)
    gt = tb.remove_left_kidney(gt)

    f1 = tb.eval_surf(final_result, gt)
    print(f1)

        
            
            
