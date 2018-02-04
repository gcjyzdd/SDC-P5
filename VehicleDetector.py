from VDT_v1 import *
from scipy.ndimage.measurements import label


class VehicleDetector():
    def __init__(self, model_file, numbuf=5, thresh=1):
        # Heat map
        self.heatMap = None
        # Number of buffers
        self.NumBuf = numbuf
        # Threshold of heat map
        self.threshold = thresh
        # Store all detected boxes
        self.allBox = []
        # Current detected boxes
        self.detectedBox = None
        # Initialization flag
        self.Initialized = False

        # Load SVC model file
        dist_pickle = pickle.load(open(model_file, "rb"))  # svc_pickle_complete
        # Store SVC model and parameters as members
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]
        self.color_space = dist_pickle["color_space"]

    @profile
    def detect(self, img):
        """Detect vehicles on an image

        :img input image
        :return the input image with boxes
        """
        if not self.Initialized:
            self.heatMap = np.zeros_like(img[:, :, 0]).astype(np.float)

        box_list = classifyImg(img, self.svc, self.X_scaler, self.color_space,
                               self.orient, self.pix_per_cell, self.cell_per_block,
                               self.spatial_size, self.hist_bins)

        # Push back the detected box_list
        if len(self.allBox) < self.NumBuf:
            self.allBox.append(box_list)
        else:
            self.allBox.pop(0)
            self.allBox.append(box_list)

        # Accumulate heat
        self.addheat()

        # Apply threshold to help remove false positives
        self.heatMap = apply_threshold(self.heatMap, self.threshold)

        # Visualize the heatmap when displaying
        self.heatMap = np.clip(self.heatMap, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(self.heatMap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        return draw_img

    def addheat(self):
        """Accumulate heat"""
        for bbox_list in self.allBox:
            # Iterate through list of bboxes
            for box in bbox_list:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                self.heatMap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
