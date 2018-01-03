import numpy as np
from os import listdir
from PIL import Image
import keras
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split


class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, data_dir, label_dir, num_classes, feature_sizes,
                 num_aspect_ratios, batch_size=32, test_size=0.2):
        'Initialization'
        self.all_image_files = np.asarray([data_dir+x for x in listdir(data_dir)])
        self.label_files = []
        for image_name in self.all_image_files:
            image_file = image_name.split('/')[-1]
            image_id = image_file.split('.')[0]
            self.label_files.append(label_dir+image_id+".npz")
        self.all_label_files = np.asarray(self.label_files)
        print self.all_label_files.shape
        self.train_images, self.val_images, self.train_labels, self.val_labels = train_test_split(self.all_image_files,
                                                                                                  self.all_label_files,
                                                                                                  test_size=test_size)

        self.batch_size = batch_size
        self.feature_sizes = feature_sizes
        self.num_classes = num_classes
        self.num_aspect_ratios = num_aspect_ratios
        self.IMG_WIDTH = 224
        self.IMG_HEIGHT = 224

    def generate(self, train=True):
        'Generates batches of samples'
        if train:
            image_files = self.train_images
            label_files = self.train_labels
        else:
            image_files = self.val_images
            label_files = self.val_labels
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = np.random.randint(len(label_files), size=self.batch_size, dtype=np.int32)
            # print indexes.shape
            # print list(indexes)
            # x = list(indexes)
            # print type(indexes[0])
            X = self.read_images(image_files[indexes])
            y = self.read_labels(label_files[indexes])

            yield preprocess_input(X), y
            # yield X, y

    def read_images(self, image_paths):
        X = np.zeros((len(image_paths), self.IMG_WIDTH, self.IMG_HEIGHT, 3))
        for i, path in enumerate(image_paths):
            img = Image.open(path)
            img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
            X[i] = img

        return X

    def read_labels(self, label_paths):
        y = {}
        for f in self.feature_sizes:
            y[str(f)] = np.zeros((self.batch_size, f, f, self.num_aspect_ratios*self.num_classes))
        for i, path in enumerate(label_paths):
            y_i = np.load(path)
            for k in y_i:
                # pad with zeros because keras wants y_true and y_pred to be of the same shape
                y[k][i, :, :, :self.num_aspect_ratios] = y_i[k]
                # label_shape = y_i[k].shape
                # padded_label = np.zeros(list(label_shape)[:-1] + [self.num_aspect_ratios*self.num_classes], dtype=np.int32)
                # padded_label[:, :, :self.num_aspect_ratios] = y_i[k]
                # out[k] = padded_label
            # y.append(out)

        return y




