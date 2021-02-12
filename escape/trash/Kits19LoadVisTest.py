import sys; sys.path.append('/home/kitamiya/Documents/python/snake3D/src')
import sys; sys.path.append('/home/kitamiya/Documents/python/snake3D/src/Toolbox')
import Toolbox as tb
from starter_code.utils import load_case
import numpy as np

img = tb.load_test_medical_image()
# img = img.transpose(1,0,2)
print(img.shape)

# img = np.random.rand(100,200,300)
tb.show_ct_image(img)