from skimage import morphology
from scipy import ndimage as ndi
from skimage.viewer import ImageViewer
from skimage.viewer import CollectionViewer
import numpy as np

im3d = np.random.rand(100, 100, 100)

viewer3d = CollectionViewer(im3d)
viewer3d.show()