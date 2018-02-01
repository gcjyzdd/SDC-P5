import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt
import time
import glob

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import logging
import argparse
from pathlib import Path
import csv

logger = logging.getLogger("VDT_v1")
logger.setLevel(logging.INFO)

# create the logging file handler
fh = logging.FileHandler("./data/VDT_v1.log")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add handler to logger object
logger.addHandler(fh)


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
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
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
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


def extract_features(imgs, ext='jpeg', color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
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
        feature = single_img_features(image, color_space='RGB', spatial_size=spatial_size,
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
    """Get features for training
    :cars image list of cars
    :notcars image list of non-cars
    """
    t = time.time()
    car_features = extract_features(cars, ext='png', color_space=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                    spatial_feat=True, hist_feat=True, hog_feat=True)

    notcar_features = extract_features(notcars, ext='png', color_space=color_space, spatial_size=spatial_size,
                                       hist_bins=hist_bins, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
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

    if 1:
        dist_pickle = {}
        dist_pickle["svc"] = svc
        dist_pickle["scaler"] = X_scaler
        dist_pickle["orient"] = orient
        dist_pickle["pix_per_cell"] = pix_per_cell
        dist_pickle["cell_per_block"] = cell_per_block
        dist_pickle["spatial_size"] = spatial_size
        dist_pickle["hist_bins"] = hist_bins
        localtime = time.localtime()
        timeString = time.strftime("%Y_%m_%d_%H_%M_%S", localtime)
        pickle.dump(dist_pickle, open("./data/svc_pickle_complete_" + timeString + ".p", "wb"))

        file_path = "./data/train_records.csv"
        my_log_file = Path(file_path)
        if my_log_file.is_file():
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


def main():
    parser = argparse.ArgumentParser(description='Train SVC using HOG features.')
    parser.add_argument('-b', action="store", dest="b")
    parser.add_argument('-c', action="store", dest="c", type=int)

    # Divide up into cars and notcars
    # images = glob.glob('*.jpeg')
    cars = glob.glob('./data/vehicles/*/*.png')
    notcars = glob.glob('./data/non-vehicles/*/*.png')
    # images.extend(glob.glob('../data/non-vehicles_smallset/*/*.jpeg'))

    data_info = data_look(cars, notcars)
    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])

    for orient in [9, 10, 12]:
        pix_per_cell = 8
        cell_per_block = 2
        for color_space in ['YCrCb', 'RGB', 'HSL', 'HSV']:
            hist_bins = 32
            spatial_size = 32
            for hog_channel in [0, 1, 2, 'ALL']:
                for kk_ in range(3):
                    trainSVC(cars, notcars, orient=orient,
                             pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             color_space=color_space, hist_bins=hist_bins,
                             spatial_size=(spatial_size, spatial_size),
                             hog_channel=hog_channel)


if __name__ == "__main__":
    main()
