import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label


########################################################
####  Section: Feature extraction                  #####
########################################################
def extract_features(imgs, spatial_size, hist_bins, cspace='BGR', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        # image = mpimg.imread(file)
        image = cv2.imread(file)

        feature = getNormalizedFeatures(image, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace = cspace, hog_channel = hog_channel)
        
        # Append the new feature vector to the features list
        features.append(feature)
    # Return list of feature vectors
    return features
    
def getNormalizedFeatures(img, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace='YCrCb', hog_channel = 'ALL'):
    ## Assume that img size is 64 x 64
    ctrans = convert_color(img, cspace=colorspace)
    ch1 = ctrans[:,:,0]
    ch2 = ctrans[:,:,1]
    ch3 = ctrans[:,:,2]
    
    vis = False
    # vis = True
    
    if hog_channel == 'ALL':
        hog1, hog_image1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis = vis, feature_vec=False)
        hog2, hog_image2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis = vis, feature_vec=False)
        hog3, hog_image3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis = vis, feature_vec=False)
        hog_feat1 = hog1.ravel() 
        hog_feat2 = hog2.ravel() 
        hog_feat3 = hog3.ravel()
        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    else:
        ch = ctrans[:,:,hog_channel]
        hog1, hog_image1 = get_hog_features(ch, orient, pix_per_cell, cell_per_block, vis = vis, feature_vec=False)
        hog_feat1 = hog1.ravel() 
        hog_features = np.hstack(hog_feat1)
    
    if vis:
        orginal_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f, axarr = plt.subplots(3, 3, figsize=(10,10))
        axarr[0, 0].set_title('Original image')
        axarr[0, 0].imshow(orginal_image)
        axarr[0, 1].set_title(colorspace + 'channel 1')
        axarr[0, 1].imshow(ch1)
        axarr[0, 2].set_title('hog_image')
        axarr[0, 2].imshow(hog_image1)
        
        axarr[1, 0].set_title('Original image')
        axarr[1, 0].imshow(orginal_image)
        axarr[1, 1].set_title(colorspace + 'channel 2')
        axarr[1, 1].imshow(ch2)
        axarr[1, 2].set_title('hog_image')
        axarr[1, 2].imshow(hog_image3)
        
        axarr[2, 0].set_title('Original image')
        axarr[2, 0].imshow(orginal_image)
        axarr[2, 1].set_title(colorspace + 'channel 3')
        axarr[2, 1].imshow(ch3)
        axarr[2, 2].set_title('hog_image')
        axarr[2, 2].imshow(hog_image3)
        plt.show()
    
    return combineFeatures(hog_features, ctrans, spatial_size, hist_bins)
    
    
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec, block_norm='L1-sqrt')
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec, block_norm='L1-sqrt')
        return features, None
    
def combineFeatures(hog_features, img, spatial_size, hist_bins):
    ## Assume that img size is 64 x 64
    # Get color features
    # spatial_features = bin_spatial(img, size=spatial_size)
    # hist_features = color_hist(img, nbins=hist_bins)
    
    # Scale features and make a prediction
    # test_features = np.hstack((spatial_features, hist_features, hog_features)).ravel()
    if spatial_size == None:
        test_features = np.hstack((hog_features)).ravel()
    else:
        if hist_bins == None:
            spatial_features = bin_spatial(img, size=spatial_size)
            test_features = np.hstack((spatial_features, hog_features)).ravel()
        else:
            spatial_features = bin_spatial(img, size=spatial_size)
            hist_features = color_hist(img, nbins=hist_bins)
            test_features = np.hstack((spatial_features, hist_features, hog_features)).ravel()
    
    return test_features

def convert_color(image, cspace='YCrCb'):
    if cspace != 'BGR':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        elif cspace == 'GRAY':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        feature_image = np.copy(image)
    return feature_image

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


########################################################
####  Section: find car using svc prediction       #####
########################################################
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cspace='YCrCb', hog_channel='ALL'):
    
    draw_img = np.copy(img)
    # img = img.astype(np.float32)/255
    
    ##### Only used for this video, to reduce the searching area.
    xstart = 640
    
    img_tosearch = img[ystart:ystop,xstart:,:]
    ctrans_tosearch = convert_color(img_tosearch, cspace=cspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    # nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step  # plus 1 to cover the rest margin
    nxsteps = nxsteps + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # For small size car, it won't appear near the bottom image according to perspective.
    nysteps = min(nysteps, 5)
    
    # Compute individual channel HOG features for the entire image
    hog1, temp = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2, temp = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3, temp = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bbox_list = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            if xpos+nblocks_per_window > hog1.shape[1]:
                print('hog1.shape', hog1.shape, 'xpos+nblocks_per_window: ', xpos+nblocks_per_window)
                xpos = hog1.shape[1] - nblocks_per_window
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feats = [hog_feat1, hog_feat2, hog_feat3]
            if hog_channel == 'ALL':
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = np.hstack(hog_feats[hog_channel])

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get combined features
            test_features = combineFeatures(hog_features, subimg, spatial_size, hist_bins)
            normalized_test_features = X_scaler.transform(test_features.reshape(1, -1))    
            test_prediction = svc.predict(normalized_test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(255,0,0),6) 
                box = ((xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart, ytop_draw+win_draw+ystart))
                bbox_list.append(box)
            # if yb == 0 or yb == 1:
                # xbox_left = np.int(xleft*scale)
                # ytop_draw = np.int(ytop*scale)
                # win_draw = np.int(window*scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,255,0),2) 
                # box = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart))
                # bbox_list.append(box)
                
    # print("scale: ", scale, 'nxblocks: ', nxblocks, 'nyblocks: ', nyblocks)
    # print('pix_per_cell: ', pix_per_cell, 'cell_per_block: ', cell_per_block, 'nblocks_per_window:', nblocks_per_window)
    # print('nxsteps: ', nxsteps, 'nysteps', nysteps)
    # draw_boxes(draw_img, bbox_list)
    # imBGRshow(draw_img)

    return draw_img, bbox_list
    
def find_cars_with_scales(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cspace='YCrCb', hog_channel='ALL'):
    boxlist = None
    for scale in scales:
        draw_img, bbox_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cspace, hog_channel)
        if boxlist == None:
            boxlist = bbox_list
        else:
            boxlist = boxlist + bbox_list
    
    draw_img = np.copy(img)
    draw_boxes(draw_img, boxlist)
    return draw_img, boxlist
    
def draw_boxes(img, bbox_list):
    for box in bbox_list:
        cv2.rectangle(img,box[0],box[1],(255,0,0), 6) 
        # cv2.rectangle(img,box[0],box[1],(0,255,0), 2) 
    return img
    
    
########################################################
####  Section: heatmap and extract car location    #####
########################################################
def getHeatmap(img, bbox_list, threshold = 1):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,bbox_list)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    return heatmap

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def imBGRshow(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    

    
########################################################
####  Section: train SVC                           #####
########################################################
def trainSVC(filename = 'svc.pickle', colorspace = 'YCrCb', orient = 9, pix_per_cell = 8, cell_per_block = 2, spatial_size = (32, 32), hist_bins = 32, hog_channel = 'ALL', sample_size = 1000):
    # Divide up into cards and notcars
    # car_images = glob.glob('train_data/vehicles_smallset/*/*.jpeg')
    car_images = glob.glob('train_data/vehicles/*/*.png')
    cars = []
    for image in car_images:
        cars.append(image)

    # notcar_images = glob.glob('train_data/not-vehicles_smallset/*/*.jpeg')
    notcar_images = glob.glob('train_data/non-vehicles/*/*.png')
    notcars = []
    for image in notcar_images:
        notcars.append(image)

    print("notcars size:", len(notcars), "cars size:", len(cars))
            
    # Reduce the sample size because HOG features are slow to compute

    if sample_size != None:
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]
        # notcars = notcars[0:sample_size + 500]

    t=time.time()
    car_features = extract_features(cars, spatial_size, hist_bins, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    notcar_features = extract_features(notcars, spatial_size, hist_bins, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)

    t2 = time.time()
    extract_features_times = round(t2-t, 2)
    print(round(t2-t, 2), 'Seconds to extract features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    print(X.shape)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
        
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    
    # Check the prediction time for a single sample
    t = time.time()
    test_accuracy = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', test_accuracy)
    print('For these',len(X_test), 'accuracy: ', test_accuracy)
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', len(X_test),'labels with SVC')

    
    summary = 'colorspace: ' + colorspace + ' orients: ' + str(orient) + ' hog_channel: ' + str(hog_channel) + ' spatial_size: ' + str(spatial_size) + ' extract feature times:' + str(extract_features_times) + ' test_accuracy: ' + str(test_accuracy)
    print(summary, filename)
    # Save classifer data on file.
    dict = {'svc': svc, 'scaler': X_scaler, 'orient': orient, 'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block, 'spatial_size': spatial_size, 'hist_bins': hist_bins, 'hog_channel': hog_channel, 'summary': summary}
    with open(filename, 'wb') as f:
        pickle.dump(dict, f)