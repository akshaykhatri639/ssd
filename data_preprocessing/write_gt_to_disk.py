import numpy as np
from os import listdir
from ground_truth import GTBuilder
from tqdm import tqdm

classes = ["background", "person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus",
               "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
num_classes = len(classes)
class_to_index = {key: value for (key, value) in zip(classes, range(num_classes))}

feature_sizes = [28, 14]
num_scales = len(feature_sizes)
aspect_ratios = [1, 2, 3, 1 / 2.0, 1 / 3.0]

save_dir = '../data/VOCdevkit/VOC2012/Preprocessed/'
label_dir = '../data/VOCdevkit/VOC2012/Annotations/'
label_files = listdir(label_dir)

gt_builder = GTBuilder(feature_sizes, aspect_ratios, num_scales, class_to_index)

for label_file in tqdm(label_files):
    # print label_file
    gt_maps = gt_builder.build_gt(label_dir+label_file)
    np.savez(save_dir+label_file.split(".")[0], **{str(key): value for (key, value) in zip(feature_sizes, gt_maps)})
    # break
