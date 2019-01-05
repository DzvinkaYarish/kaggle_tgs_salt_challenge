import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
import glob
import torch
from torch.utils import data
import os
from dataset.dataset import TGSSaltDataset
from losses import iou, lovasz_hinge, StableBCELoss
from models.unet_vgg import UNet11
from torchvision import transforms as torch_transforms
# from models.unet_resnet import UNetResNet
# from models.unet_densenet import UNetDenseNet
# from models.unet_seresnext import UNetSEResNext
from models.universal_UNet import UNet
import numpy as np
from losses import iou_binary
from torch.nn import functional as F
from sklearn.metrics import jaccard_similarity_score
import cv2
import pickle
import yaml
import time
import datetime
import random


CUDA = True
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')


def get_model(model_name, weight_path):
    if model_name == 'dense_unet':
        model = UNet('DenseNet', encoder_depth=201, num_classes=1)
    elif model_name == 'seresnext_unet':
        model = UNet('SeResNext', encoder_depth=50, num_classes=1)
    elif model_name == 'resnet_unet':
        model = UNet('ResNet', encoder_depth=50, num_classes=1)
    else:
        raise NotImplementedError('Not implemented model %s' % model_name)
    model.load_state_dict(torch.load(weight_path)['model'])
    model = model.eval()
    if CUDA:
        return model.cuda()
    return model


def get_data(dataroot, test_path):
    depths_df = pd.read_csv(os.path.join(dataroot, 'train/train.csv'))
    file_list = list(depths_df['id'].values)

    num_to_select = len(file_list) // 10
    files_for_validation = random.sample(file_list, num_to_select)

    dataset_val = TGSSaltDataset(dataroot, files_for_validation, augment=False)  # using all files for validation

    test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
    test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]

    return dataset_val, test_file_list


def predict_with_tta(models, image):
    y_pred = torch.zeros([1, 1, 128, 128])
    if not np.sum(image):
        return y_pred.data.numpy()[0, 0, :, :]

    y_pred_flipped = torch.zeros([1, 1, 128, 128])
    if CUDA:
        y_pred = y_pred.cuda()
        y_pred_flipped = y_pred_flipped.cuda()
    norm_transforms = torch_transforms.Compose([
        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32'))
    image_tensor = norm_transforms(image_tensor).unsqueeze(0)
    image_tensor_f = torch.from_numpy(np.transpose(np.fliplr(image), (2, 0, 1)).astype('float32'))
    image_tensor_f = norm_transforms(image_tensor_f).unsqueeze(0)
    if CUDA:
        image_tensor = image_tensor.cuda()
        image_tensor_f = image_tensor_f.cuda()
    for model in models:
        y_pred += torch.sigmoid(model(image_tensor))
        y_pred_flipped += torch.sigmoid(model(image_tensor_f))
    res_pred = y_pred.cpu().data.numpy()[0, 0, :, :] + np.fliplr(y_pred_flipped.cpu().data.numpy()[0, 0, :, :]) # WHY [0, 0, :, :]
    return res_pred / (2 * len(models))


def test_on_val(models, val_dataset):
    val_predictions = []
    val_masks = []
    for image, mask in tqdm(data.DataLoader(val_dataset, batch_size=30)):
        pred_shape = list(image.shape)
        pred_shape[1] = 1  # grayscale image channels
        y_pred = torch.zeros(pred_shape)
        if CUDA:
            y_pred = y_pred.cuda()
        image = image.type(torch.FloatTensor)
        if CUDA:
            image = image.cuda()
        image = Variable(image)
        for model in models:
            y_pred += torch.sigmoid(model(image))

        val_predictions.append(y_pred.cpu().data.numpy() / len(models))  # ToDo: [0, 0, :, :]??
        val_masks.append(mask)

    val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]

    val_masks_stacked = np.vstack(val_masks)[:, 0, :, :]

    val_binary_prediction = (val_predictions_stacked > 0.5).astype(int)

    iou_values = []
    for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
        # iou_v = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
        iou_v = iou_binary(y_mask, p_mask)
        iou_values.append(iou_v)
    iou_values = np.array(iou_values)

    accuracies = [
        np.mean(iou_values > iou_threshold)
        for iou_threshold in np.linspace(0.5, 0.95, 10)
    ]
    print('Iou metric: %.3f' % (np.mean(accuracies)))


def test_submit(test_path, models, test_file_list):
    all_predictions = []
    soft_predictions = []
    tq = tqdm(total=len(test_file_list), desc='test')
    with torch.no_grad():
        for file in test_file_list:
            image_folder = os.path.join(test_path, "images")
            image_path = os.path.join(image_folder, file + ".png")
            image = cv2.imread(str(image_path))
            image = image / 255.0
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            pred = predict_with_tta(models=models, image=image)
            pred = cv2.resize(pred, (101, 101), interpolation=cv2.INTER_LANCZOS4)
            soft_predictions.append(pred)
            all_predictions.append((pred > 0.5).astype(int))
            tq.update(1)
        tq.close()

    def rle_encoding(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    all_masks = []
    for p_mask in all_predictions:
        p_mask = rle_encoding(p_mask)
        all_masks.append(' '.join(map(str, p_mask)))

    submit = pd.DataFrame([test_file_list, all_masks]).T
    submit.columns = ['id', 'rle_mask']
    sub_name = config['experiment_desc']
    submit.to_csv(f'experiments/default_experiment/{sub_name}_{st}.csv', index=False)

    if config['serialize_pred']:
        with open(f'{sub_name}_{st}.pickle', 'wb') as file:
            pickle.dump(soft_predictions, file)


if __name__ == '__main__':
    with open('configs/salt_segmentator_k_folds_test.yaml', 'r') as f:
        config = yaml.load(f)

    # models_indexes = config['indexes_in_ensemble']
    # models = []
    # for model_index in models_indexes:
    #     model_class = config[f'ensemble_{model_index}']['class']
    #     model_snapshot = config[f'ensemble_{model_index}']['snapshot']
    #     print(f'Using [{model_class}] from [{model_snapshot}]')
    #     models.append(get_model(model_class, model_snapshot))

    val_dataset, test_file_list = get_data(config['dataroot'], config['test_path'])

    test_on_val(models, val_dataset)
    test_submit(config['test_path'], models, test_file_list)
