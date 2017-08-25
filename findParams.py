import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from utils import *


# ### Step 1: Train svc
# # Parameters tuning.
# # colorspace = 'YCrCb'
# colorspaces = ['YUV', 'YCrCb'] 
# # orient = 9
# orients = [9, 11, 13]
# pix_per_cell = 8
# cell_per_block = 2
# # spatial_size = (32, 32)
# spatial_sizes = [(24, 24), (32, 32)]
# hist_bins = 32
# hog_channels = ['ALL', 0]
# # hog_channel = 0
# sample_size = 8000

# for spatial_size in spatial_sizes:
    # for colorspace in colorspaces:
        # for orient in orients:
            # for hog_channel in hog_channels:
                # filename = colorspace + str(orient) + str(hog_channel) + str(spatial_size) + '.pickle'
                # trainSVC(filename , colorspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, sample_size)
                
                
                
                
### Step 1: Train svc
# Parameters tuning.
# colorspace = 'YCrCb'
colorspaces = ['HSV', 'YUV', 'YCrCb'] 
# orient = 9
orients = [13]
pix_per_cell = 8
cell_per_block = 2
# spatial_size = (32, 32)
spatial_sizes = [(16, 16)]
hist_bins = 64
hog_channels = ['ALL']
sample_size = None

for spatial_size in spatial_sizes:
    for colorspace in colorspaces:
        for orient in orients:
            for hog_channel in hog_channels:
                filename = colorspace + str(orient) + str(hog_channel) + str(spatial_size) + '.pickle'
                trainSVC(filename , colorspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, sample_size)