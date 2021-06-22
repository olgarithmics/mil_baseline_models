"""Pytorch Dataset object that loads 27x27 patches that contain single cells."""

import os
import random

import numpy as np

import cv2

from data_aug import hematoxylin_eosin_aug


class BreastCancerDataset(object):

    def __init__(self,patch_size, shuffle_bag,seed=None, augmentation=True, **kwargs):


        self.patch_size = patch_size
        self.augmentation = augmentation
        self.seed = seed
        self.stride=16
        self.shuffle_bag=shuffle_bag
        super(BreastCancerDataset, self).__init__(**kwargs)

    def fd_hu_moments(self, image):
        image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()

        return feature.reshape(-1)


    def mean_std(self, image):
        pixel_num = (image.shape[0] * image.shape[1] * 3)
        channel_sum = np.sum(image, axis=(0, 1))
        channel_sum_squared = np.sum(np.square(image), axis=(0, 1))
        bgr_mean = channel_sum / pixel_num
        bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
        rgb_mean = list(bgr_mean)[::-1]
        rgb_std = list(bgr_std)[::-1]

        return np.concatenate([rgb_mean, rgb_std], axis=0).reshape(-1)



    def load_bags(self, wsi_paths):


        bag_list=[]
        labels_list=[]
        for ibag, path in enumerate(wsi_paths):

            basename=os.path.basename(path)
            img_name = os.path.splitext(basename)[0]
            temp_name=os.path.join(os.path.dirname(path),"img{}.txt".format(img_name))
            label = 0 if "benign" in img_name else 1
            img =cv2.imread(path)

            img = hematoxylin_eosin_aug(img)

            cropped_cells = []

            with open(temp_name, "r") as cell_loc:
                lines = cell_loc.readlines()

                for line in lines:
                    x = line.split(",")[0]
                    y = line.split(",")[1]
                    patch = img[int(x) - self.stride:int(x) + self.stride,
                            int(y) - self.stride:int(y) + self.stride]

                    patch = np.asarray(patch, dtype=np.float32)
                    global_feature = np.hstack([self.mean_std(patch), self.fd_hu_moments(patch)])

                    cropped_cells.append([global_feature])

                if label == 1:
                    curr_label = 1
                else:
                    curr_label = -1
                stack_img = np.concatenate(cropped_cells, axis=0)

                bag_list.append((stack_img))
                labels_list.append((curr_label))

        return np.asarray(bag_list), np.asarray(labels_list, dtype=np.float32)

