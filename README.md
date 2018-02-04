# SDC-P5
Self-Driving-Car Project 5: Vehicle Detection and Tracking

## Intro

Below are the description of files.

**Report**: 

  * [report.md](./report.md), the brief report of this project.

IPython notebook: 

  * [./VDT_v1.ipynb](VDT_v1.ipynb), proceeding this project step by step.

Python script:

  * [VDT_v1.py](./VDT_v1.py), a collection of functions used in other scripts.
  * [trainSVC.py](./trainSVC.py), train a linear SVC with a bunch of parameters and save the model with time stamp
  * [testSVC.py](./testSVC.py), test the trained SVC with images.
  * [VehicleDetector](./VehicleDetector.py), a `VehicleDetector` class that performs vehicle detections.

Folders:

  * The [result](./result) folder: contains the SVC model file and a `csv` file of comparison of different combinations of parameters.

  * The [output_images](./output_images): output of test images and the project video.

## Run

Train a SVC:

```sh
python trainSVC.py
```

Test the SVC

```sh
python testSVC -im image.jpg -svc svc_model.p
```

To run the video pipeline, open the IPython notebook and follow the instructions.