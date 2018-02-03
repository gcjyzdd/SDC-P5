from VDT_v1 import *
import sys


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
                    sys.exit()


if __name__ == "__main__":
    main()
