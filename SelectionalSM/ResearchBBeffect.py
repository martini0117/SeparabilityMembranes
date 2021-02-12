import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/ellipsoid_fit')
import math

import matplotlib

from geomdl.exchange import import_json
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
from geomdl.exchange import export_json
from sklearn.metrics import f1_score


import os

import Toolbox as tb
import ellipsoid_fit as ef

np.random.seed(seed=173)

#160,181,189
# cases = range(150,190)
# cases = range(150,160)
cases = 150
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/contours/kidney_EN/'
img_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'

#輪郭ごとの分離度の大小関係が，分離度を計算する領域の位置や大きさによって変わるかを調べる

window_center = 0
window_width = 300

case_str = tb.get_full_case_id(cases)
print(case_str)

#ボリュームデータの読み込み
nii0 = nib.load(img_path + case_str + '/imaging.nii.gz')
img = nii0.get_data()
img = tb.window_function(img,window_center,window_width)

nii0=nib.load(img_path + case_str + '/segmentation.nii.gz')
gt = nii0.get_data().astype(np.uint8)
gt = np.where(gt==2,0,gt)
gt = tb.remove_left_kidney(gt)

result_path = data_path + case_str + '/result'
max_count = np.zeros((10,))

for i in range(100):
    wide = np.array([30 + np.random.randint(-5,5),50 + np.random.randint(-20,20),50 + np.random.randint(-20,20)])
    center = np.array([50, 280, 330]) + np.random.randint(-5,5,(3,))
    rangem = center - wide
    rangep = center + wide
    bb = [slice(rangem[0],rangep[0]),slice(rangem[1],rangep[1]),slice(rangem[2],rangep[2])]

    print(wide)

    separabilities = []
    for f in os.listdir(result_path):
        result = import_json(result_path + '/' + f)[0]
        separability = tb.separability_via_bb(result, img, bb)
        separabilities.append(separability)

    max_count[np.argmax(separabilities)] += 1

print(max_count)
np.savetxt('BBeffect.txt', max_count)





            
            
