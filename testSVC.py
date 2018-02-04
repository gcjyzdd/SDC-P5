from VDT_v1 import *
from scipy.ndimage.measurements import label


def main():
    parser = argparse.ArgumentParser(description='Test Linear SVC.')
    parser.add_argument('-im', '--im_path', help='Input the image path', required=True)
    parser.add_argument('-svc', '--svc_path',
                        default='./result/svc_pickle_complete_v2_2018_02_04_05_06_04.p',
                        help='Path of the SVC model file')
    parser.add_argument('-save_path', '--save_path',
                        help='Path to save the output image')

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

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 0)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    out_img = draw_boxes(img, box_list)
    plt.imshow(out_img)
    if args['save_path']:
        plt.savefig(args['save_path'], bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    if __name__ == '__main__':
        main()
