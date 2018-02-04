# SDC Project 5: Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cell 1-3 of the IPython notebook [file](./VDT_v1.ipynb)(and in lines 105 through 124 of the file called [VDT_v1.py](./VDT_v1.py)). 

In the IPython notebook, I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

<div style="text-align:center"><img width=100% src ='./output_images/car_notcar.jpg' /></div>

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


<div style="text-align:center"><img width=100% src ='./output_images/features.jpg' /></div>


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and selected the combination with the highest performance:

1. I trained the linear SVC with the script [trainLSVC.py](./trainLSVC.py)
2. The overall result is saved as a `csv` [file](./result/train_records_v2.csv)

Sort the accuracy scores and I got the final choice of HOG parameters:

* `color_space`: `LUV`
* `orient`: 12
* `pix_per_cell`: 8
* `cell_per_block`: 2
* `hist_bins`: 32
* `spatial_size`: 32

The test accuracy of this combination is **0.992**.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `vehicle` and `non-vehicle` datasets. The cooresponding code is line 272-334 in [VDT_v1.py](./VDT_v1.py). 

* I used HOG features along with `LUV` color features and spatial features because more features can give the model more freedom to fit the data. 
* The extracted features are scaled using a `StandardScaler`. 
* The dataset are splitted using `train_test_split` and 20% is used for cross validation.

The script [trainLSVC.py](./trainLSVC.py) shows the actual implementation of training.

To train a linear SVC, run the script:

```sh
python trainLSVC.py
```

To test the SVC with an image, I have provieded a script called [testSVC.py](./testSVC.py). Run:

```
python testSVC.py -im test_img.jpg
```

to test an image called `test_img.jpg`. Run `python testSVC.py -h` to see more details.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

<div style="text-align:center"><img width=100% src ='./examples/sliding_windows.jpg' /></div>


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

<div style="text-align:center"><img width=100% src ='./examples/sliding_window.jpg' /></div>

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

<div style="text-align:center"><img width=100% src ='./examples/bboxes_and_heat.png' /></div>


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
<div style="text-align:center"><img width=100% src ='./examples/labels_map.png' /></div>

### Here the resulting bounding boxes are drawn onto the last frame in the series:
<div style="text-align:center"><img width=100% src ='./examples/output_bboxes.png' /></div>



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

