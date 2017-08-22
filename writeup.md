##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./examples/notcars_train_data.JPG
[image2]: ./examples/cars_train_data.JPG
[image3]: ./examples/hog_feature_example.png
[image4]: ./examples/not_car_hog_feature_example.png
[image5]: ./examples/slidingWindow1.png
[image6]: ./examples/slidingWindow1p5.png
[image7]: ./examples/slidingWindow2.png
[image8]: ./examples/slidingWindow2p5.png
[image9]: ./examples/slidingWindow3.png
[image10]: ./examples/slidingWindow3p5.png
[image11]: ./examples/slidingWindow4.png


[image20]: ./examples/output_bboxes.png
[image21]: ./examples/output_bboxes1.png
[image22]: ./examples/output_bboxes2.png
[image23]: ./examples/output_bboxes3.png
[image24]: ./examples/output_bboxes4.png
[image25]: ./examples/output_bboxes5.png

[image30]: ./examples/labels_map.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the Section: Feature extraction of the file called `utils.py`.  Please check function get_hog_features, which mainly employ hog function from skimage.feature package

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

​										Non-vehicle images

![alt text][image1]

​											Vehicle images

![alt text][image2]



I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `spatial_size`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=14`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]
![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and 

| colorspace | orientations  | hog_channel | spatial_size | Extract Time | Test_Accuracy | Features |
| YCrCb |  9  | ALL | None | 520.98s | 0.9878 | 5292 |
| YCrCb |  11 | ALL | None |  604.28 | 0.9906 | 6468 |
| YCrCb |  13 | ALL | None |  695.82 | 0.9894 | 7644 |
| YCrCb |  9  | 0 | None | 312.69 | 0.9569 | 1764 |
| YCrCb |  11  | 0 | None | 384.47 | 0.9556 | 2156 |
| YCrCb |  13 | 0 | None | 441.13 | 0.9569 | 2548 |
| YUV |  9  | ALL | None | 571.96 | 0.9884 | Accurate | 5292 |
| YUV |  11 | ALL | None | 724.37s | 0.9881 | 6468 |
| YUV |  13 | ALL | None | 819.37s | 0.9909 | 7644 |
| YUV |  9  | 0 | None | 370.23s | 0.9512 | | 1764 |
| YUV |  11 | 0 | None | 435.03 | 0.9547 | 2156 |
| YUV |  13 | 0 | None | 542.9 | 0.9525 | 2548 |

| YCrCb |  9  | ALL | (16, 16) | 72.4 | 0.9894 | 6060 |
| YCrCb |  11 | ALL | (16, 16) | 74.16 | 0.9934 | 7236 | 
| YCrCb |  13 | ALL | (16, 16) |  75.65 | 0.9953 | 8412 |
| YCrCb |  9  | 0 | (16, 16) | 171.85 | 0.9734 | 2532 |
| YCrCb |  11  | 0 | (16, 16) | 205.71 | 0.9759 | 2924 |
| YCrCb |  13 | 0 | (16, 16) | 303.78 | 0.9778 | 3316 |
| YUV |  9  | ALL | (16, 16) | 72.4 | 0.9928 | 6060 |
| YUV |  11 | ALL | (16, 16) | 75.31s | 0.9941 | 7236 |
| YUV |  13 | ALL | (16, 16) | 75.64 | 0.9938 | 8412 |
| YUV |  9  | 0 | (16, 16) | 172.67 | 0.9778 | 2532 |
| YUV |  11 | 0 | (16, 16) | 206.42 | 0.9772 | 2924 |
| YUV |  13 | 0 | (16, 16) | 320.44 | 0.9756 | 3316 |

| YUV |  9  | ALL | (24, 24) | 71.74 | 0.9925 | 7020 |
| YUV |  9  | 0 | (24, 24) | 229.24 | 0.9788 | | 3492 |
| YUV |  11 | ALL | (24, 24) | 74.39s | 0.9938 | 8196 |
| YUV |  11 | 0 | (24, 24) | 263.58 | 0.98 | 3884 |
| YUV |  13 | ALL | (24, 24) | 75.8 | 0.9953 | 9372 |
| YUV |  13 | 0 | (24, 24) | 317.79 | 0.9769 | 4276 |
| YCrCb |  9  | ALL | (24, 24) | 74.32 | 0.9928 | 7020 |
| YCrCb |  9  | 0 | (24, 24) | 169.96 | 0.9703 | 3492 |
| YCrCb |  11 | ALL | (24, 24) | 74.48s | 0.9944 | 8196 |
| YCrCb |  11 | 0 | (24, 24) | 201.75 | 0.9812 | 3884 |
| YCrCb |  13 | ALL | (24, 24) | 75.56 | 0.995 | 9372 |
| YCrCb |  13 | 0 | (24, 24) | 319.83 | 0.9822 | 4276 |

| YUV |  9  | ALL | (32, 32) | 72.86 | 0.9903 | 8364 |
| YUV |  9  | 0 | (32, 32) | 172.87 | 0.9778 | 4836 |
| YUV |  11 | ALL | (32, 32) | 75.44s | 0.9928 | 9540 |
| YUV |  11 | 0 | (32, 32) | 268.26 | 0.9819 | 5228 |
| YUV |  13 | ALL | (32, 32) | 76.22 | 0.9959 | 10716 |
| YUV |  13 | 0 | (32, 32) | 330.6 | 0.9788 | 5620 |
| YCrCb |  9  | ALL | (32, 32) | 73.82 | 0.9938 | 8364 |
| YCrCb |  9  | 0 | (32, 32) | 171.51 | 0.9797 | 4836 |
| YCrCb |  11 | ALL | (32, 32) | 74.6s | 0.9931 | 9540 |
| YCrCb |  11 | 0 | (32, 32) | 208.36 | 0.9794 | 5228 |
| YCrCb |  13 | ALL | (32, 32) | 77.27 | 0.9928 | 10716 |
| YCrCb |  13 | 0 | (32, 32) | 331.2 | 0.9812 | 5620 |

From the above parameters exploration, it is found that colorspace YUV, orientenation = 13, All channel for hog_feature extraction, and (32, 32) spatial size got highest accuracy -- test accuracy 0.9959. 

All trained SVC are saved in folder -- trainedSVC.   findParameters.py was used to do this parameters exploration.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the default classifier parameters and using HOG features and spatial intensity with size (32, 32) (I did not use channel intensity histogram features) and was able to achieve a test accuracy of 99.59%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section titled "Section: find car using svc prediction" of file utils.py. I adapted the method find_cars from the lesson materials. The method combines HOG feature extraction for a certain scale with a sliding window search, but rather than perform with different sliding window size.  Since object is bigger when closer, we need to use different sliding window size. It means differnt scales.  In the same section, function find_cars_with_scales is defined, which will search image with different scale -- meaning different scale window.


Scale: 1

![alt text][image5]

Scale: 1.5

![alt text][image6]

Scale: 2

![alt text][image7]

Scale: 2.5

![alt text][image8]

Scale: 3

![alt text][image9]

Scale: 3.5

![alt text][image10]

Scale: 4

![alt text][image11]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are one frame and its corresponding heatmap:

![alt text][image25]
![alt text][image30]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

From the output video, it can be found that there are still have some false positives. In my opinion, there are two ways to make it more robust.

1. Needs more training data.  From output_video, it can be found that false positives are mainly related with area with tree shadow.  By increasing more training data related with this category could make classifier more robust.
2. Could smooth over contiguous frames.  It is based on assumption, that one vehicle won't appear/disappear suddenly in contiguous frames.  Thus, we can filter false positive if one detected car location show in one frame, but disappear in next frame.

The training data is not general enough, or not suit for this project video.  The reason is that the trained classifier can reach high test accuracy, around 99%, but it will mis-predict in video frames with sliding window.   

