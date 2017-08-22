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

# filename = 'YCrCb13ALL(16, 16).pickle'
# filename = 'YUV13ALL(24, 24).pickle'
filename = 'YUV13ALL(24, 24).pickle'
# filename = 'svc.pickle'
# filename = 'YUV13ALL(32, 32)32.pickle'

### Step 1: Train svc
# Parameters tuning.
# colorspace = 'YCrCb'
colorspace = 'YUV'
# orient = 9
orient = 15
pix_per_cell = 8
# pix_per_cell = 12
cell_per_block = 2
spatial_size = (32, 32)
# spatial_size = (16, 16)
hist_bins = 32
hog_channel = 'ALL'
# hog_channel = 0
sample_size = None

# trainSVC(filename , colorspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, sample_size)

### Step 2: Find cars in one test image
# Load trained classifer from file.
dist_pickle = pickle.load( open(filename, "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
hog_channel = dist_pickle["hog_channel"]

print('SVC summary: ', dist_pickle["summary"])

img = cv2.imread('test_images/test5.jpg')
ystart = 380
ystop = 720
# scale = 1.5

# out_img, bbox_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)

# scales = [1.5, 2.5, 3.5, 4.5]
# scales = [1, 1.5, 2, 2.5, 3, 3.5, 4]
scales = [1.5, 2.5, 3.5]
out_img, bbox_list = find_cars_with_scales(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)

imBGRshow(out_img)

# # Test all test images
# test_images = glob.glob('test_images/*.jpg')
# for test_image in test_images:
    # img = cv2.imread(test_image)
    # # out_img, bbox_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
    # out_img, bbox_list = find_cars_with_scales(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
    # imBGRshow(out_img)

## Step 3: Get heat image
threshold = 4
heatmap = getHeatmap(img, bbox_list, threshold = threshold)

from scipy.ndimage.measurements import label
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(img), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
plt.show()


### Step 4: define process_image function
from scipy.ndimage.measurements import label
def process_image(img):
    ystart = 380
    ystop = 720
    # scales = [1, 1.5, 2, 2.5, 3, 3.5, 4]

    out_img, bbox_list = find_cars_with_scales(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
    
    heatmap = getHeatmap(img, bbox_list, threshold = threshold)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
    
    
### Step 5: Generate video output
from moviepy.editor import VideoFileClip

output_filename = 'project_video_output.mp4'
# output_filename = 'test_video_output.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,2)
clip1 = VideoFileClip("project_video.mp4")
# clip1 = VideoFileClip("test_video.mp4")
write_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
write_clip.write_videofile(output_filename, audio=False)


