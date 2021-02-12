import glob
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import os

data = np.zeros((500,256,256,3),dtype=np.uint8)

# for i in range(data.shape[0]):
#     radius = random.randint(10,30)
#     center = (random.randint(40,220),random.randint(40,220))
#     cv2.circle(data[i,:,:,:], center, radius, (128, 128, 128),thickness=-1)


data += np.random.randint(0,100,data.shape,dtype=np.uint8)

output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/sample/'

os.makedirs(output_path + 'kidney/0/',exist_ok=True)
os.makedirs(output_path + 'notkidney/0/',exist_ok=True)

for i in range(data.shape[0]):
    pil_img = Image.fromarray(data[i,:,:,:])
    pil_img.save(output_path + '/notkidney/0/' + str(i) + '.png')
