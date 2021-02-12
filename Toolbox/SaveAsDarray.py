import sys; sys.path.append('/home/kitamiya/Documents/python/snake3D/src')
sys.path.append('/home/kitamiya/Documents/python/snake3D/src/Toolbox')
import Toolbox as tb
from starter_code.utils import load_case
import numpy as np

num = int(sys.argv[1])

volume, segmentation = load_case(num)
img = np.asarray(volume.dataobj)
print(img.shape)

np.save("/home/kitamiya/Documents/python/snake3D/data/numpy/case" + str(num) + ".npy",img)