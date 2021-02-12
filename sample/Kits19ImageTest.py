import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src')
sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import Toolbox as tb
from starter_code.utils import load_case
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from mayavi import mlab

volume, segmentation = load_case(123)
print(type(segmentation))

seg = np.asarray(segmentation.dataobj)
seg = np.where(seg == 1,1,0)
seg = seg.astype(np.float32)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(seg[100:180,260:340,300:380])
# plt.show()

mlab.clf()
mlab.contour3d(seg[50:250,200:400,250:450])
mlab.show()
