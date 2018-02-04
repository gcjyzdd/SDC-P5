import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt
import time
import glob
from functools import wraps
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import logging
import argparse
from pathlib import Path
import csv


### define profilers (https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module)
PROF_DATA = {}


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling


def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print("Function %s called %d times. " % (fname, data[0]))
        print('Execution time max: %.3f, average: %.3f' % (max_time, avg_time))


def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}

### end of profiler

logger = logging.getLogger("VDT_v1")
logger.setLevel(logging.INFO)

# create the logging file handler
fh = logging.FileHandler("./data/VDT_v1.log")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add handler to logger object
logger.addHandler(fh)


# Define a function that takes an image, a list of bounding boxes,
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn

    for i in range(len(bboxes)):
        cv2.rectangle(draw_img,bboxes[i][0], bboxes[i][1], color=color,
                      thickness=thick)
    return draw_img


def convert_color(img, color_space='RGB'):
    feature_image = None
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    return feature_image


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec,
                                  block_norm='L2-Hys')
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec,
                       block_norm='L2-Hys')
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space)
    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def extract_features(imgs, hog_channel, ext='jpeg', color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # If using PNG images, scale it to 0-255 from 0-1
        if ext == 'png':
            image = image * 255
            image = image.astype(np.uint8)
        feature = single_img_features(image, color_space=color_space, spatial_size=spatial_size,
                                      hist_bins=hist_bins, orient=orient,
                                      pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                      spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        features.append(feature)
    return features


# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    tmp_im = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = tmp_im.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = tmp_im.dtype

    # Return data_dict
    return data_dict


def getFeaturesFromImages(cars, notcars, orient=9,
                          pix_per_cell=8,
                          cell_per_block=2,
                          color_space='YCrCb',
                          hist_bins=32,
                          spatial_size=(32, 32),
                          hog_channel='ALL'):
    """Get features of list of images for training
    :cars image list of cars
    :notcars image list of non-cars
    :returns X_train, y_train, X_test, y_test, X_scaler
    """
    t = time.time()
    car_features = extract_features(cars, hog_channel, ext='png', color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    spatial_feat=True, hist_feat=True, hog_feat=True)

    notcar_features = extract_features(notcars, hog_channel, ext='png', color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       spatial_feat=True, hist_feat=True, hog_feat=True)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    return X_train, y_train, X_test, y_test, X_scaler


def trainSVC(cars, notcars, orient=9,
             pix_per_cell=8,
             cell_per_block=2,
             color_space='YCrCb',
             hist_bins=32,
             spatial_size=(32, 32),
             hog_channel='ALL'):
    X_train, y_train, X_test, y_test, X_scaler = getFeaturesFromImages(cars, notcars, orient=orient,
                                                                       pix_per_cell=pix_per_cell,
                                                                       cell_per_block=cell_per_block,
                                                                       color_space=color_space, hist_bins=hist_bins,
                                                                       spatial_size=spatial_size,
                                                                       hog_channel=hog_channel)
    """Train a linear SVC using labeled images.
    This function saves the trained model to file with time stamp.
    
    Input params:
    :cars list of car images
    :notcars list of not-car images
    """
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    acc = svc.score(X_test, y_test)
    print('Test Accuracy of SVC = ', round(acc, 4))

    # Save the model file with time stamp
    if 1:
        dist_pickle = {}
        dist_pickle["svc"] = svc
        dist_pickle["scaler"] = X_scaler
        dist_pickle["orient"] = orient
        dist_pickle["pix_per_cell"] = pix_per_cell
        dist_pickle["cell_per_block"] = cell_per_block
        dist_pickle["spatial_size"] = spatial_size
        dist_pickle["hist_bins"] = hist_bins
        dist_pickle["color_space"] = color_space
        localtime = time.localtime()
        timeString = time.strftime("%Y_%m_%d_%H_%M_%S", localtime)
        pickle.dump(dist_pickle, open("./result/svc_pickle_complete_v2_" + timeString + ".p", "wb"))

        file_path = "./result/train_records_v2.csv"
        my_result_file = Path(file_path)
        if my_result_file.is_file():
            # file exists
            with open(file_path, 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow([timeString, orient, pix_per_cell, cell_per_block, color_space,
                                     hist_bins, spatial_size[0], hog_channel, acc])
        else:
            with open(file_path, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow(
                    ['TimeStamp', 'orient', 'pix_per_cell', 'cell_per_block', 'color_space', 'hist_bins',
                     'spatial_size', 'hog_channel', 'Acc'])
                spamwriter.writerow([timeString, orient, pix_per_cell, cell_per_block, color_space,
                                     hist_bins, spatial_size[0], hog_channel, acc])

    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


@profile
def find_cars(img, color_space,
              xstart, ystart, xstop, ystop,
              scale, svc, X_scaler,
              orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins):
    """Find cars in an image
    :return list of boxes
    """

    box_list = []

    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)  # RGB2YCrCb
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                box_list.append([(xbox_left + xstart, ytop_draw + ystart),
                                 (xbox_left + xstart + win_draw, ytop_draw + win_draw + ystart)])
    return box_list  # draw_img


def classifyImg(img, svc, X_scaler, color_space,
         orient, pix_per_cell, cell_per_block,
         spatial_size, hist_bins):

    xstart = 0
    ystart = 400
    xstop = img.shape[1]
    ystop = 656
    scale = 3

    box_list = find_cars(img, color_space, xstart, ystart, xstop, ystop,
                         scale, svc, X_scaler,
                         orient, pix_per_cell, cell_per_block, spatial_size,
                         hist_bins)

    xstart = 200
    ystart = 400
    xstop = 1100
    ystop = 600
    scale = 2

    box_list2 = find_cars(img, color_space, xstart, ystart, xstop, ystop,
                          scale, svc, X_scaler,
                          orient, pix_per_cell, cell_per_block, spatial_size,
                          hist_bins)

    box_list.extend(box_list2)

    xstart = 0
    ystart = 400
    xstop = img.shape[1]
    ystop = 500
    scale = 1

    box_list2 = find_cars(img, color_space, xstart, ystart, xstop, ystop,
                          scale, svc, X_scaler,
                          orient, pix_per_cell, cell_per_block, spatial_size,
                          hist_bins)

    box_list.extend(box_list2)

    xstart = 620
    ystart = 405
    xstop = 880
    ystop = 460
    scale = 0.5

    box_list2 = find_cars(img, color_space, xstart, ystart, xstop, ystop,
                          scale, svc, X_scaler,
                          orient, pix_per_cell, cell_per_block, spatial_size,
                          hist_bins)

    box_list.extend(box_list2)

    return box_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img
