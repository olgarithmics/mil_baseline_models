
from bag_loader import ColonCancerDataset
from breast_bag_loader import BreastCancerDataset

import misvm
import numpy as np
from sklearn.metrics import precision_score, recall_score

from dataset import load_files

def model_training(classifier, dataset):
    train_bags = dataset['train']
    test_bags = dataset['test']

    # test_images, test_labels = ColonCancerDataset(patch_size=27, augmentation=False).load_bags(test_bags)
    #
    # train_images, train_labels = ColonCancerDataset(patch_size=27, augmentation=True).load_bags(train_bags)

    test_images, test_labels = BreastCancerDataset(patch_size=32, shuffle_bag=False,augmentation=False).load_bags(test_bags)

    train_images, train_labels = BreastCancerDataset(patch_size=32, shuffle_bag=True,augmentation=True).load_bags(train_bags)

    classifier.fit(train_images, train_labels)
    predictions = classifier.predict(test_images)

    y_pred = np.sign(predictions)
    y_true = test_labels
    accuracy = np.average(y_true == np.sign(predictions))

    print("accuracy {}".format(accuracy))

    #     auc = roc_auc_score(y_true, y_pred)
    #     print("AUC {}".format(auc))

    y_true = list(map(lambda x: 0 if x == -1 else 1, y_true))
    y_pred = list(map(lambda x: 0 if x == -1 else 1, y_pred))
    precision = precision_score(y_true, np.round(np.clip(y_pred, 0, 1)))
    print("precision {}".format(precision))

    recall = recall_score(y_true, np.round(np.clip(y_pred, 0, 1)))
    print("recall {}".format(recall))


    return accuracy,recall, precision

if __name__ == '__main__':
    n_folds=4
    run=5
    data_path="Breast_Cancer_Cells"

    acc = np.zeros((run, n_folds), dtype=float)
    precision = np.zeros((run, n_folds), dtype=float)
    recall = np.zeros((run, n_folds), dtype=float)

    classifiers = {}
    classifiers['MISVM'] =misvm.MISVM(kernel='linear', C=1.0, max_iters=50)
    # classifiers['miSVM'] =  misvm.miSVM(kernel='linear', C=1.0, max_iters=50)
    # classifiers['MissSVM'] =misvm.MissSVM(kernel='linear', C=1.0, max_iters=50)
    for algorithm, classifier in classifiers.items():
        for irun in range(run):

            acc = np.zeros((run, n_folds), dtype=float)
            precision = np.zeros((run, n_folds), dtype=float)
            recall = np.zeros((run, n_folds), dtype=float)
            datasets = load_files(dataset_path=data_path,n_folds=n_folds, rand_state=irun,ext=".tif")

            for ifold in range(n_folds):
                print('run=', irun, '  fold=', ifold)

                acc[irun][ifold], recall[irun][ifold], precision[irun][ifold]= \
                model_training(classifier,datasets[ifold])

        print("algorithm",algorithm)
        print('mi-net mean accuracy = ', np.mean(acc))
        print('std = ', np.std(acc))
        print('mi-net mean precision = ', np.mean(precision))
        print('std = ', np.std(precision))
        print('mi-net mean recall = ', np.mean(recall))
        print('std = ', np.std(recall))
