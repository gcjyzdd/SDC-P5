from VDT_v1 import *


def main():
    parser = argparse.ArgumentParser(description='Test Linear SVC.')
    parser.add_argument('-im', '--im_path', help='Input the image path', required=True)
    parser.add_argument('-svc', '--svc_path',
                        default='./result/svc_pickle_complete_v2_2018_02_03_15_19_24.p',
                        help='Path of the SVC model file')

    args = vars(parser.parse_args())

    img = mpimg.imread(args['im_path'])

    dist_pickle = pickle.load(open(args['svc_path'], "rb"))  # svc_pickle_complete

    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    color_space = dist_pickle["color_space"]

    box_list = classifyImg(img, svc, X_scaler, color_space,
                orient, pix_per_cell, cell_per_block,
                spatial_size, hist_bins)

    out_img = draw_boxes(img, box_list)
    plt.imshow(out_img)
    plt.show()


if __name__ == "__main__":
    if __name__ == '__main__':
        main()
