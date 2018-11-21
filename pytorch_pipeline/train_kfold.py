import pandas as pd
import torch
import os
import glob
from dataset.dataset import TGSSaltDataset
from models.universal_UNet import UNet
import numpy as np
from trainer import Trainer
import cv2
from sklearn.model_selection import StratifiedKFold
import yaml
import time
import logging
import datetime


CONFIG_FILENAME = 'configs/salt_segmentator.yaml'
CUDA = True


def init_logger(config):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    logfile = f'exp_{config["experiment_desc"]}_{st}.log'
    if not os.path.exists('logs'):
            os.mkdir('logs')
    logging.basicConfig(filename=os.path.join('logs', logfile),
                        level=logging.DEBUG)


def get_data(data_root, file_list_train, file_list_val, pseudolabels=False):
    test_files = []
    if pseudolabels:
        test_path = os.path.join(data_root, 'test')
        test_files = glob.glob(test_path + "/masks/*.png")

    dataset = TGSSaltDataset(data_root, file_list_train, test_file_list=test_files)
    dataset_val = TGSSaltDataset(data_root, file_list_val, augment=False)
    return dataset, dataset_val


def calculate_coverage_classes_inplace(train_df, config):
    st_coverage = time.time()

    masks_path = os.path.join(config['train_path'], 'masks')
    train_df['coverage'] = train_df['id'].map(
        lambda x:
        np.sum(cv2.imread(filename=(os.path.join(masks_path, x) + '.png'), flags=cv2.IMREAD_GRAYSCALE) / 255.0) /
        np.prod(config['input_image_size'])
    )  # getting coverage percents for masks

    def cov_to_class(val):
        for i in range(config['num_coverage_classes']):
            if val * (config['num_coverage_classes'] - 1) <= i:
                return i

    train_df["coverage_class"] = train_df['coverage'].map(cov_to_class)

    coverage_calc_time = time.time() - st_coverage
    print(f'coverage_calculated in {coverage_calc_time} sec')
    return train_df


def get_train_df(config):
    depths_df = pd.read_csv(config['depth_file'])
    if config['recalculate_coverage_classes']:
        train_df = pd.read_csv(config['train_file'])
        train_df = calculate_coverage_classes_inplace(train_df, config)
    else:
        train_df = pd.read_csv(os.path.join(train_path, 'train_with_coverage.csv'))  # precomputed

    if config['stratify_depth'] and config['stratify_coverage']:
        depths_df.sort_values('z', inplace=True)
        depths_df['depth_fold'] = (list(range(config['num_depth_classes'])) * depths_df.shape[0])[:depths_df.shape[0]]
        train_df.merge(depths_df,  on='id')
        train_df['fold_class'] = train_df.apply(
            lambda row: f"{row['depth_fold']}_{row['coverage_class']}", axis=1)
    elif config['stratify_coverage']:
        train_df.rename(index=str, columns={"coverage_class": "fold_class"})
    elif config['stratify_depth']:
        train_df = depths_df.copy()
        train_df.rename(index=str, columns={"depth_fold": "fold_class"})
    else:
        train_df = pd.read_csv(config['train_file'])
        train_df['fold_class'] = train_df['id'].map(lambda x: 0)
    return train_df


if __name__ == '__main__':
    with open(CONFIG_FILENAME, 'r') as f:
        config = yaml.load(f)

    init_logger(config)

    train_path = config['train_path']

    num_folds = config['num_folds']

    train_df = get_train_df(config)

    skf = StratifiedKFold(n_splits=num_folds, random_state=42, shuffle=True)

    file_list = list(train_df['id'].values)

    folds_splits = list(skf.split(file_list,  train_df['fold_class']))
    folds_to_train = config['folds_to_train']  # to train on different machines

    for current_fold_number in folds_to_train:

        train_indx, val_indx = folds_splits[current_fold_number]

        stage = 0

        model = UNet('ResNet', encoder_depth=34, num_classes=1)
        if CUDA:
            model = model.cuda()

        train, val = get_data(data_root=config['dataroot'],
                              file_list_train=np.take(a=file_list, indices=train_indx),
                              file_list_val=np.take(a=file_list, indices=val_indx),
                              pseudolabels=False)

        trainer = Trainer(model, config, stage=stage, fold=current_fold_number)
        trainer.train(train, val)

        model.load_state_dict(torch.load(trainer.get_checkpoint_filename())['model'])

        stage += 1

        train, val = get_data(data_root=config['dataroot'],
                              file_list_train=np.take(a=file_list, indices=train_indx),
                              file_list_val=np.take(a=file_list, indices=val_indx),
                              pseudolabels=False)

        trainer = Trainer(model, config, stage=stage, fold=current_fold_number)
        trainer.train(train, val)

        if config['cosine_annealing']:

            model.load_state_dict(torch.load(trainer.get_checkpoint_filename()['model']))

            stage += 1
            trainer = Trainer(model, config, stage=stage, fold=current_fold_number)
            train, val = get_data(data_root=config['dataroot'],
                                  file_list_train=np.take(a=file_list, indices=train_indx),
                                  file_list_val=np.take(a=file_list, indices=val_indx),
                                  pseudolabels=False)
            trainer.train(train, val)
