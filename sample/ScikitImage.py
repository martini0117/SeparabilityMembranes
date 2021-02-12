from skimage import data, io, filters
from skimage.viewer import ImageViewer

image = data.coins()

print(type(image))

viewer = ImageViewer(image) # doctest: +SKIP
viewer.show()               # doctest: +SKIP