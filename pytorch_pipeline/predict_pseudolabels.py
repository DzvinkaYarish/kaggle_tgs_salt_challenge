import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
import glob
import torch
from torch.utils import data
import os
from dataset.dataset import TGSSaltDataset
from models.unet_vgg import UNet11
from torchvision import transforms as torch_transforms
from models.unet_resnet import UNetResNet
from models.unet_densenet import UNetDenseNet
from models.unet_seresnext import UNetSEResNext
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import jaccard_similarity_score
import cv2
import itertools
import yaml

CUDA = False

def iou(y_true, y_pred):
    _EPSILON = 1e-6
    overlap = y_pred * y_true  # Logical AND
    union = y_pred + y_true  # Logical OR

    l = (overlap.sum() + _EPSILON) / (float(union.sum()) + _EPSILON)
    return l


def get_model(model_name, weight_path):
    if model_name == 'dense_unet':
        model = UNetDenseNet(encoder_depth=201, num_classes=1)
    elif model_name == 'seresnext_unet':
        model = UNetSEResNext(encoder_depth=50, num_classes=1)
    else:
        raise NotImplementedError('Not implemented model')
    model.load_state_dict(torch.load(weight_path)['model'])
    model = model.eval()
    if CUDA:
        return model.cuda()
    return model


def get_data(test_path):
    test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
    test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]

    return test_file_list


def predict(models, image):
    norm_transforms = torch_transforms.Compose([
        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32'))
    image_tensor = norm_transforms(image_tensor).unsqueeze(0)
    if CUDA:
        image_tensor = image_tensor.cuda()
    predictions = []
    with torch.no_grad():
        for model in models:
            y_pred = torch.sigmoid(model(image_tensor)).cpu().data.numpy()[0, 0, :, :]
            predictions.append(y_pred)
            image_tensor = torch.from_numpy(np.transpose(np.fliplr(image), (2, 0, 1)).astype('float32'))
            image_tensor = norm_transforms(image_tensor).unsqueeze(0)
            y_pred_tta = torch.sigmoid(model(image_tensor)).cpu().data.numpy()[0, 0, :, :]
            predictions.append(np.fliplr(y_pred_tta))
            model.cpu()

    counter = 0
    iou_score = 0
    for a, b in itertools.combinations(predictions, 2):
        iou_score += iou(a > 0.5, b > 0.5)
        counter += 1

    return iou_score / counter, sum(predictions) / len(predictions)


def predict_pseudolabels(test_path, models, test_file_list):
    tq = tqdm(total=len(test_file_list), desc='test')
    threshold = 0.8
    good_masks = 0
    bad_masks = 0
    image_folder = os.path.join(test_path, "images")
    mask_folder = os.path.join(test_path, "masks")
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    for file in test_file_list:
        image_path = os.path.join(image_folder, file + ".png")
        image = cv2.imread(str(image_path))
        image = image / 255.0
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        score, pred_mask = predict(models=models, image=image)
        print("Score = {}".format(score))
        if score >= threshold:
            pred_mask = cv2.resize(pred_mask, (101, 101), interpolation=cv2.INTER_LANCZOS4)
            pred_mask = (pred_mask > 0.5).astype(int) * 255
            cv2.imwrite(os.path.join(mask_folder, file + ".png"), pred_mask)
            good_masks +=1
        else:
            bad_masks += 1
        tq.update(1)
    print("Good masks = {}".format(good_masks))
    print("Bas masks = {}".format(bad_masks))
    tq.close()


if __name__ == '__main__':
    with open('configs/salt_segmentator.yaml', 'r') as f:
        config = yaml.load(f)
    dense_net = get_model('dense_unet', '/home/f/tgs_salt/nets_res/densenet/net.h5')
    seresnext_net = get_model('seresnext_unet', '/home/f/tgs_salt/pytorch_source/best_1_0.h5')
    test_file_list = get_data(config['test_path'])
    predict_pseudolabels(config['test_path'], [dense_net, seresnext_net], test_file_list)
