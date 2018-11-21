import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
import glob
import torch
from torch.utils import data
import os
from dataset.dataset import TGSSaltDataset
from losses import iou, lovasz_hinge, StableBCELoss, iou_binary
from models.unet_vgg import UNet11
from torchvision import transforms as torch_transforms
from models.unet_resnet import UNetResNet
from models.unet_densenet import UNetDenseNet
from models.unet_seresnext import UNetSEResNext
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import jaccard_similarity_score
import cv2
from sklearn.model_selection import StratifiedKFold
from trainer import Trainer

DATA_PATH = '../data/'
DROP_BLACK_PICS = False
CUDA = True
COSINE_ANNEALING = False
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")


def get_model(model_name, weight_path):
    if model_name == 'dense_unet':
        model = UNetDenseNet(encoder_depth=201, num_classes=1)
    elif model_name == 'seresnext_unet':
        model = UNetSEResNext(encoder_depth=50, num_classes=1)
    else:
        raise NotImplementedError('Not implemented model')
    model.load_state_dict(torch.load(weight_path)['model'])
    model = model.eval()
    return model.cuda()

def get_data(binary=True):
    directory = '../data'
    depths_df = pd.read_csv(os.path.join(directory, 'train/train.csv'))

    train_path = os.path.join(directory, 'train')
    file_list = list(depths_df['id'].values)

    file_list_val = file_list[::10]
    dataset_val = TGSSaltDataset(directory, file_list_val, augment=False)
    test_path = os.path.join(directory, 'test')
    test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
    test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]

    return test_path, dataset_val, test_file_list

def predict_with_tta(models, image):
    y_pred = torch.zeros([1,1,128,128]).cuda()
    y_pred_flipped = torch.zeros([1, 1, 128, 128]).cuda()
    norm_transforms = torch_transforms.Compose([
        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32'))
    image_tensor = norm_transforms(image_tensor).unsqueeze(0).cuda()
    image_tensor_f = torch.from_numpy(np.transpose(np.fliplr(image), (2, 0, 1)).astype('float32'))
    image_tensor_f = norm_transforms(image_tensor_f).unsqueeze(0).cuda()
    for model in models:
        y_pred += torch.sigmoid(model(image_tensor))
        pred = torch.sigmoid(model(image_tensor_f))
        y_pred_flipped += pred
    res_pred = y_pred.cpu().data.numpy()[0, 0, :, :] + np.fliplr(y_pred_flipped.cpu().data.numpy()[0, 0, :, :])
    return res_pred / (2 * len(models))

def test_submit(val_dataset, models, test_file_list):
    all_predictions = []
    tq = tqdm(total=len(test_file_list), desc='test')
    # with torch.no_grad():
    #     for file in test_file_list:
    #         image_folder = os.path.join(test_path, "images")
    #         image_path = os.path.join(image_folder, file + ".png")
    #         mask_folder = os.path.join(test_path, "masks")
    #         mask_path = os.path.join(mask_folder, file + ".png")
    #         image = cv2.imread(str(image_path))
    #         image = image / 255.0
    #         image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    #
    #
    #
    #         pred = predict_with_tta(models=models, image=image)
    #         pred = cv2.resize(pred, (101, 101), interpolation=cv2.INTER_LANCZOS4)
    #
    #         mask = cv2.imread(str(mask_path))
    #         mask = mask / 255.0
    #         mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    #
    #         all_predictions.append((pred > 0.5).astype(int))
    #         tq.update(1)
    #     tq.close()

    # iou_score = iou_binary(all_predictions, mask.to('cuda'), ignore=255, per_image=True)

    val_predictions = []
    val_masks = []
    model = models[0]
    for image, mask  in tqdm(data.DataLoader(val_dataset, batch_size=30)):
        image = Variable(image.type(torch.FloatTensor).cuda())
        y_pred = torch.sigmoid(model(image)).cpu().data.numpy()
        # y_pred = torch.sigmoid(model(image)).cpu().data.numpy() # TODO predict_with_tta(models=models, image=image)
        val_predictions.append(y_pred)
        val_masks.append(mask)

    val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]

    val_masks_stacked = np.vstack(val_masks)[:, 0, :, :]




    metric_by_threshold = []
    threshold = 0.5
    val_binary_prediction = (val_predictions_stacked > threshold).astype(int)

    iou_values = []
    it = 0
    iou_v = iou_binary(val_masks_stacked, val_binary_prediction)
    if iou_v < 95:
        print('bad')
        print(f'{it} [{iou_v}]')
        iou_values.append(iou_v)
    iou_values = np.array(iou_values)
    # for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
    #     # iou_v = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
    #     iou_v = iou_binary(y_mask.flatten(), p_mask.flatten())
    #     if iou_v < 95:
    #         print('bad')
    #         # plt.imshow(y_mask)
    #         # plt.show()
    #         # plt.imshow(p_mask)
    #         # plt.show()
    #         # for mask in y_mask:
    #         #     plt.imshow(mask)
    #         #     plt.show()
    #         # plt.imshow(y_mask.flatten())
    #     print(f'{it} [{iou_v}]')
    #     iou_values.append(iou_v)
    # iou_values = np.array(iou_values)

    # iou_threshold = 0.5
    # accuracies = [
    #     np.mean(iou_values > iou_threshold)
    #     for iou_threshold in np.linspace(0.5, 0.95, 10)
    # ]
    print('Threshold: %.1f, Metric: %.3f' % ( 0.5 , np.mean(iou_values)))
    return np.mean(iou_values)
    # for threshold in np.linspace(0, 1, 21):
    #     val_binary_prediction = (val_predictions_stacked > threshold).astype(int)
    #
    #     iou_values = []
    #     for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
    #         iou_v = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
    #         iou_values.append(iou_v)
    #     iou_values = np.array(iou_values)
    #
    #     accuracies = [
    #         np.mean(iou_values > iou_threshold)
    #         for iou_threshold in np.linspace(0.5, 0.95, 10)
    #     ]
    #     print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))
    #     metric_by_threshold.append((np.mean(accuracies), threshold))
    #
    # def rle_encoding(x):
    #     dots = np.where(x.T.flatten() == 1)[0]
    #     run_lengths = []
    #     prev = -2
    #     for b in dots:
    #         if (b > prev + 1): run_lengths.extend((b + 1, 0))
    #         run_lengths[-1] += 1
    #         prev = b
    #     return run_lengths
    #
    # all_masks = []
    # for p_mask in all_predictions:
    #     p_mask = rle_encoding(p_mask)
    #     all_masks.append(' '.join(map(str, p_mask)))
    # submit = pd.DataFrame([test_file_list, all_masks]).T
    # submit.columns = ['id', 'rle_mask']
    # submit.to_csv('submit_snapshot_ensemble.csv', index=False)

if __name__ == '__main__':
    stage = 2

    train_path = os.path.join(DATA_PATH, 'train')
    depths_df = pd.read_csv(os.path.join(train_path, 'train_with_coverage.csv'))

    number_of_folds = 5
    depths = pd.read_csv(os.path.join(DATA_PATH, 'depths.csv'))

    depths.sort_values('z', inplace=True)
    depths['depth_fold'] = (list(range(number_of_folds)) * depths.shape[0])[:depths.shape[0]]
    merged_df = pd.merge(depths_df, depths, on='id')

    merged_df['fold_class'] = merged_df.apply(
        lambda row: str(row['depth_fold']) + '_' + str(row['coverage_class']), axis=1)

    skf = StratifiedKFold(n_splits=number_of_folds, random_state=42, shuffle=True)

    if DROP_BLACK_PICS:
        black_indxs = pd.read_csv(os.path.join(train_path, 'black_imgs.csv'))
        merged_df = merged_df[~merged_df['id'].isin(black_indxs['id'])]

    file_list = list(merged_df['id'].values)

    # val_df = merged_df[merged_df['id'].isin(file_list_val)]
    # sns.distplot(merged_df.z, label="Train")
    # sns.distplot(val_df.z, label="Test")
    # plt.legend()
    # plt.title("Depth distribution")
    # plt.show()

    fold = 0
    folds_iou = []
    for train_indx, val_indx in skf.split(file_list, merged_df.fold_class):
        # Plotting the depth distributions¶

        file_list_train = [file for i, file in enumerate(file_list) if i in train_indx]
        file_list_val = [file for i, file in enumerate(file_list) if i in val_indx]

        # if fold == 0:
        #     fold += 1
        #     continue

        dataset_val = TGSSaltDataset(DATA_PATH, file_list_val, augment=False)
        # test, val, file_list = get_data()
        # seresnext_net1 = get_model('seresnext_unet', '../pytorch_source/snapshots/snapshot_2_0_0.h5')
        seresnext_net0 = get_model('seresnext_unet', '../pytorch_source/snapshots/best_1_0.h5')
        # seresnext_net2 = get_model('seresnext_unet', '../pytorch_source/snapshots/snapshot_2_0_1.h5')
        # seresnext_net3 = get_model('seresnext_unet', '../pytorch_source/snapshots/snapshot_2_0_2.h5')
        # seresnext_net4 = get_model('seresnext_unet', '../pytorch_source/snapshots/snapshot_2_0_3.h5')
        # loss_fn = get_lossfn('focal')
        # trainer = Trainer(model, loss_fn, stage, fold=fold)
        # trainer = Trainer(seresnext_net0, None, -1, fold=-1)
        # trainer._validate()

        # test_submit(test, dataset_val, [seresnext_net0], file_list)
        folds_iou.append(test_submit(dataset_val, [seresnext_net0], []))
    print(folds_iou)
