import glob
import random
import re
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from data_aug import hematoxylin_eosin_aug, normalize, random_flip_img,random_rotate_img
import os

def load_files(dataset_path,n_folds,rand_state,ext):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """
    paths=[]
    for root, dirs, image_paths in os.walk(dataset_path, topdown=False):

        for image_path in image_paths:
            if image_path.endswith(ext):
                paths.append(os.path.join(root, image_path))


    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
    datasets = []
    for train_idx, test_idx in kf.split(paths):
        dataset = {}
        dataset['train'] = [paths[ibag] for ibag in train_idx]
        dataset['test'] = [paths[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets


def load_dataset(dataset_path, n_folds, rand_state):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """

    pos_path = glob.glob(dataset_path + '/1/img*')

    neg_path = glob.glob(dataset_path + '/0/img*')

    all_path = pos_path + neg_path

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
    datasets = []
    for train_idx, test_idx in kf.split(all_path):
        dataset = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets



def generate_batch(path, train, ext, normalization=True):
    """
    Generate the bags of instances
    Parameters
    ----------
    path : list of the file paths of the bags: [colon_cancer_patches/0/img1,
    colon_cancer_patches/0/img3,....]

    Returns
    -------
    bags: list of tuples(np.ndarray,np.ndarray,str). Each tuple  contains an np.ndarray
    of the patches of each image, the label of the image
    and a list consisting of the filenames of each patch, respectively

    """
    bags = []


    for each_path in path:
        name_img = []
        img = []
        img_path = glob.glob(each_path + '/*.{}'.format(ext))

        num_ins = len(img_path)

        label = int(each_path.split('/')[-2])

        if label == 1:
            curr_label = np.ones(num_ins, dtype=np.float32)
        else:
            curr_label = np.zeros(num_ins, dtype=np.float32)
        for each_img in img_path:
            img_data = np.asarray(Image.open(each_img), dtype=np.float32)

            if train:
                img_data = hematoxylin_eosin_aug(img_data)
                img_data = random_flip_img(img_data, horizontal_chance=0.5, vertical_chance=0.5)
                img_data = random_rotate_img(img_data)
            if normalization:
                img_data = normalize(img_data)
                img_data = np.asarray(img_data, dtype=np.float32)
                img_data /= 255


            img.append(np.expand_dims(img_data, 0))
            name_img.append(each_img.split('/')[-1])
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, curr_label, name_img))

    return bags


def Get_train_valid_Path(Train_set, ifold, train_percentage=0.9):
    indexes = np.arange(len(Train_set))
    np.random.seed(ifold)
    random.shuffle(indexes)

    num_train = int(train_percentage * len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val


def load_config_file(nfile, abspath=False):

    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)


def get_coordinates(path):
    coords = (int(re.findall("\d{1,3}", path.split("-")[1])[0]),
              int(re.findall("\d{1,3}", path.split("-")[2])[0]))
    return coords


def exclude_self(Idx):
    new_Idx = np.empty((Idx.shape[0], Idx.shape[1] - 1))
    for i in range(Idx.shape[0]):
        try:
            new_Idx[i] = Idx[i, Idx[i] != i][:Idx.shape[1] - 1]
        except Exception as e:
            print(Idx[i, ...], new_Idx.shape, Idx.shape)
            raise e
    return new_Idx.astype(np.int)



