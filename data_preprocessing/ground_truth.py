import math
import numpy as np
from data_loader import Bbox, read_labels_from_xml
from data_loader import IMG_SIZE
from os import listdir
from PIL import Image

class GTBuilder():

    def __init__(self, feature_sizes, aspect_ratios, num_scales, class_to_index, min_scale=0.2,
                 max_scale=0.9, IoUthreshold=0.5):
        self.feature_sizes = feature_sizes
        self.aspect_ratios = aspect_ratios
        self.num_scales = num_scales
        self.scales = self.get_scales(min_scale, max_scale)
        # print self.scales
        self.IoUthreshold = IoUthreshold
        self.class_to_index = class_to_index

    def get_scales(self, min_scale, max_scale):
        step_size = (max_scale - min_scale) / (self.num_scales - 1)
        scales = [min_scale + step_size * i for i in range(self.num_scales)]
        return scales

    def build_dense_ground_truth(self, feature_size, scale, gt_boxes):
        op = np.zeros((feature_size, feature_size, len(self.aspect_ratios)+1))
        for i in range(feature_size):
            for j in range(feature_size):
                box_centre = ((i + 0.5) / feature_size * IMG_SIZE, (j + 0.5) / feature_size * IMG_SIZE)

                # for all boxes of this size centred at box_center
                for k, ar in enumerate(self.aspect_ratios):
g
                    # Box centre along with this width and height forms one default box
                    width = scale*math.sqrt(ar)*IMG_SIZE
                    height = scale/math.sqrt(ar)*IMG_SIZE
                    # print scale, math.sqrt(ar), IMG_SIZE, "width", width

                    bbox = Bbox(None, 0, box_centre[0], box_centre[1], width, height)
                    # compare this box to a ground_truth box
                    for gt_box in gt_boxes:
                        IoU = bbox.compute_IoU(gt_box)
                        if IoU > self.IoUthreshold:
                            op[i, j, k] = self.class_to_index[gt_box.object_type]
                            # op[i, j, k] = IoU

                # the last default box of aspect ratio 1 but bigger size
                k += 1
                ar = math.sqrt(scale*(scale+0.1))
                width = ar * IMG_SIZE
                height = ar * IMG_SIZE

                bbox = Bbox(None, 0, box_centre[0], box_centre[1], width, height)
                # compare this box to a ground_truth box
                for gt_box in gt_boxes:
                    if bbox.compute_IoU(gt_box) > self.IoUthreshold:
                        # set class to the class of the ground_truth box
                        # op[i, j, k] = self.class_to_index[gt_box.object_type]
                        op[i, j, k] = bbox.compute_IoU(gt_box)

        return op

    def build_gt_all_sizes(self, gt_boxes):
        gt_maps = []

        for i, f in enumerate(self.feature_sizes):
            gt_map = self.build_dense_ground_truth(f, self.scales[i], gt_boxes)
            gt_maps.append(gt_map)

        return gt_maps

    def build_gt(self, label_file):
        gt_boxes = read_labels_from_xml(label_file)

        # print [gt_box.__dict__ for gt_box in gt_boxes]
        gt_maps = self.build_gt_all_sizes(gt_boxes)

        return gt_maps

    def index_to_bbox(self, i, j, k, feature_index):
        feature_size = self.feature_sizes[feature_index]
        box_centre = ((i + 0.5) / feature_size * IMG_SIZE, (j + 0.5) / feature_size * IMG_SIZE)

        scale = self.scales[feature_index]
        if k<5:
            ar = self.aspect_ratios[k]
            width = scale * math.sqrt(ar) * IMG_SIZE
            height = scale / math.sqrt(ar) * IMG_SIZE
        else:
            ar = math.sqrt(scale * (scale + 0.1))
            width = ar * IMG_SIZE
            height = ar * IMG_SIZE

        return Bbox(0, 0, box_centre[0], box_centre[1], width, height)


if __name__ == "__main__":
    classes = ["background", "person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus",
               "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
    num_classes = len(classes)
    class_to_index = {key: value for (key, value) in zip(classes, range(num_classes))}

    feature_sizes = [20, 16, 12, 8, 4]
    num_scales = len(feature_sizes)
    aspect_ratios = [1, 2, 3, 1 / 2.0, 1 / 3.0]

    label_dir = '../data/VOCdevkit/VOC2012/Annotations/'
    label_files = listdir(label_dir)
    label_file = '../data/VOCdevkit/VOC2012/Annotations/2007_000027.xml'

    gt_builder = GTBuilder(feature_sizes, aspect_ratios, num_scales, class_to_index)
    gt_maps = gt_builder.build_gt(label_file)

    print num_classes
    for gt_map in gt_maps:
        print gt_map.shape
        print np.where(gt_map != 0)

    # image_file = '../data/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg'
    # img = Image.open(image_file)

    save_dir = '../data/VOCdevkit/VOC2012/Preprocessed/2007_000027'
    np.savez(save_dir, **{str(key): value for (key, value) in zip(feature_sizes, gt_maps)})







