# ssd
Implementation of Single Shot Multibox Detector

# Simplifications made over the paper
- Instead of selecting the top 3N boxes to include in the loss where N is the number of ground truth boxes for this sample,
we select top k where k is fixed across all samples. This makes it necessary to do a hyperparameter search over k but simplifies the implementation, atleast in tensorflow

# Todo:
- Find a setting of k that works well
- Implement evaluation metrics (mAP). Currently just working with precision and recall by treating it as classification over the default boxes
- Train on bigger dataset, add data augmentation
- Regress from of default box locations to make the predicted bounding boxes tighter.
