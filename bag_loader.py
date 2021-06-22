import os
import numpy as np
import scipy.io
import cv2
import glob
from PIL import Image
from data_aug import hematoxylin_eosin_aug, normalize


class ColonCancerDataset(object):

    def __init__(self, patch_size, seed=None, augmentation=True, **kwargs):

        self.patch_size = patch_size
        self.augmentation = augmentation
        self.seed = seed
        super(ColonCancerDataset, self).__init__(**kwargs)

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

    def preprocess_bags(self, wsi_paths):
        wsi = []

        for each_path in wsi_paths:

            mat_files = glob.glob(os.path.join(os.path.split(each_path)[0], "*epithelial.mat"))

            epithelial_file = scipy.io.loadmat(mat_files[0])
            label = 0 if (epithelial_file["detection"].size == 0) else 1

            img_data = np.asarray(Image.open(each_path), dtype=np.float32)

            if self.augmentation:
                img_data = hematoxylin_eosin_aug(img_data, seed=self.seed)
            wsi.append((img_data.astype(np.float32), label, each_path))

        return wsi

    def load_bags(self, wsi_paths):
        wsi = self.preprocess_bags(wsi_paths)

        bags = []
        labels = []
        for ibag, bag in enumerate(wsi):

            num_ins = 0
            img = []

            for enum, cell_type in enumerate(['epithelial', 'fibroblast', 'inflammatory', 'others']):

                dir_cell = os.path.splitext(bag[2])[0] + '_' + cell_type + '.mat'

                with open(dir_cell, 'rb') as f:
                    mat_cell = scipy.io.loadmat(f)

                    num_ins += len(mat_cell['detection'])
                    for (x, y) in mat_cell['detection']:
                        x = np.round(x)
                        y = np.round(y)

                        if x < np.floor(self.patch_size / 2):
                            x_start = 0
                            x_end = self.patch_size
                        elif x > 500 - np.ceil(self.patch_size / 2):
                            x_start = 500 - self.patch_size
                            x_end = 500
                        else:
                            x_start = x - np.floor(self.patch_size / 2)
                            x_end = x + np.ceil(self.patch_size / 2)
                        if y < np.floor(self.patch_size / 2):
                            y_start = 0
                            y_end = self.patch_size
                        elif y > 500 - np.ceil(self.patch_size / 2):
                            y_start = 500 - self.patch_size
                            y_end = 500
                        else:
                            y_start = y - np.floor(self.patch_size / 2)
                            y_end = y + np.ceil(self.patch_size / 2)

                        patch = bag[0][int(y_start):int(y_end), int(x_start):int(x_end)]

                        patch = normalize(patch)
                        patch = np.asarray(patch, dtype=np.float32)
                        # patch /= 255

                        global_feature = np.hstack([self.mean_std(patch), self.fd_hu_moments(patch)])

                        img.append([global_feature])



            if bag[1] == 1:
                curr_label = 1
            else:
                curr_label = -1

            stack_img = np.concatenate(img, axis=0)

            assert num_ins == len(stack_img)
            #             m_samples = stack_img.shape[0]
            #             stack_img = stack_img.reshape(m_samples, -1)

            bags.append((stack_img))
            labels.append((curr_label))
        return np.asarray(bags), np.asarray(labels, dtype=np.float32)






