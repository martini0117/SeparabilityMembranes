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

import Toolbox as tb
import ellipsoid_fit as ef

# cases = range(210,299)
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'

cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/RemovedW5Data/'

# cases = range(210,299)
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'

S_seg = 0
S_con_seg = 0


for case in cases:
    case_str = tb.get_full_case_id(case)

    nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()

    nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
    gt = nii0.get_data().astype(np.uint8)

    nii0=nib.load(data_path + case_str + '/prediction.nii.gz')
    seg = nii0.get_data().astype(np.uint8)

    nii1 = nib.load(output_path + case_str + '/prediction_contour.nii.gz')
    con_seg = nii1.get_data()

    # tb.show_ct_image(gt)
    # tb.show_ct_image(seg)
    # tb.show_ct_image(con_seg)

    eval_seg = tb.evaluate(case,seg)
    eval_con_seg = tb.evaluate(case,con_seg)

    ave_eval_seg = np.mean(eval_seg)
    ave_eval_con_seg = np.mean(eval_con_seg)

    print(case_str)
    print(eval_seg)
    print(eval_con_seg)
    print(ave_eval_seg)
    print(ave_eval_con_seg)

    S_seg += ave_eval_seg
    S_con_seg += ave_eval_con_seg

print(S_seg/len(cases))
print(S_con_seg/len(cases))



