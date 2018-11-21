import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm
import glob
import torch
from torch.nn import functional as F
from torch.utils import data
from losses import iou_binary, iou_metric, iou, lovasz, iou_loss, binary_xloss, focal
import numpy as np
import logging
import time
from schedulers import WarmRestart
import os
# from tensorboardX import SummaryWriter





# writer = SummaryWriter('runs')

# logging.basicConfig(filename='train.log',level=logging.DEBUG)
#

# bce as metric at validation?
# saving snapshots?


REPORT_EACH = 8
CUDA=True

class Trainer(object):
    def __init__(self, model, configs, fold=0, stage=0):
        self.model = model
        self.stage = stage
        self.fold = fold
        self.config = configs

        self.checkpoint_filename = f'best_stg{self.stage}_fld{self.fold}.h5'
        self.checkpoint_path = os.path.join('experiments', self.config['experiment_desc'])
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)


    def get_checkpoint_filename(self):
        return os.path.join(self.checkpoint_path, self.checkpoint_filename)

    def _get_optimizer(self):
        opt_name = self.config['stage%d' % self.stage]['optimizer']['name']
        if opt_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['stage%d' % self.stage]['optimizer']['lr'])
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['stage%d' % self.stage]['optimizer']['lr'])
        elif opt_name == 'adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.config['stage%d' % self.stage]['optimizer']['lr'])
        else:
            raise ValueError('Optimizer [%s] not recognized.' % opt_name)
        return optimizer

    def _get_loss(self):
        loss_name = self.config['stage%d' % self.stage]['loss']
        if loss_name == 'bce':
            loss = binary_xloss
        elif loss_name == 'lovasz':
            loss = lovasz
        elif loss_name == 'focal':
            loss = focal
        else:
            raise ValueError("Loss [%s] not recognized." % loss_name)
        return loss

    def _get_scheduler(self, optimizer):
        scheduler_name = self.config['stage%d' % self.stage]['scheduler']['name']
        if scheduler_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode=self.config['stage%d' % self.stage]['scheduler']['mode'],
                                                              patience=self.config['stage%d' % self.stage]['scheduler']['patience'],
                                                              factor=self.config['stage%d' % self.stage]['scheduler']['factor'],
                                                              min_lr=self.config['stage%d' % self.stage]['scheduler']['min_lr'],
                                                              eps=self.config['stage%d' % self.stage]['scheduler']['epsilon'])
        elif scheduler_name == 'warmrestart':
            scheduler = WarmRestart(optimizer,
                                    T_max=self.config['stage%d' % self.stage]['scheduler']['epochs_per_cycle'],
                                    eta_min=self.config['stage%d' % self.stage]['scheduler']['min_lr'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % scheduler_name)
        return scheduler

    def _init_params(self):
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler(self.optimizer)
        self.loss_fn = self._get_loss()

    def train(self, train_data, val_data):
        self._init_params()
        epochs = self.config['stage%d' % self.stage]['num_epochs']
        self.best_metric = 0
        for epoch in range(0, epochs + 1):
            train_loss, train_iou = self._run_epoch(epoch, train_data)
            val_loss, val_iou = self._validate(val_data)
            self.scheduler.step(val_iou)

            if val_iou > self.best_metric:
                self.best_metric = val_iou

                print(f'updating {self.checkpoint_filename}')
                torch.save({
                    'model': self.model.state_dict()
                }, self.get_checkpoint_filename())

            print("Train [loss: %.3f, iou: %.3f] , Val [loss: %.3f, iou: %.3f, best: %.3f]\n" % (
                train_loss, train_iou, val_loss, val_iou, self.best_metric))

            logging.debug("Epoch: %d, Val: %.3f, Val IoU: %.3f" % (
                epoch, val_loss, val_iou))

    def _run_epoch(self, epoch, train_data):
        self.model = self.model.train()
        losses = []
        metrics = []
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        t = tqdm(data.DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True))
        t.set_description('Epoch {}, lr {}'.format(epoch, lr))
        for data_batch_ditct in t:
            image = data_batch_ditct['image']
            mask = data_batch_ditct['mask']
            image = image.type(torch.FloatTensor)
            if CUDA:
                image = image.cuda()
            y_pred = self.model(Variable(image))
            if CUDA:
                loss = self.loss_fn(y_pred, mask.cuda())
            else:
                loss = self.loss_fn(y_pred, mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data[0])

            preds = (y_pred.data > 0).float()
            if CUDA:
                metrics.append(iou_binary(preds, mask.to('cuda'), ignore=255, per_image=True))
            else:
                metrics.append(iou_binary(preds, mask, ignore=255, per_image=True))
            t.set_postfix(loss=np.mean(losses[-REPORT_EACH:]),
                          iou=np.mean(metrics[-REPORT_EACH:]))
        train_loss = np.mean(losses)
        train_iou = np.mean(metrics)
        logging.debug("Epoch: %d, Train: %.3f, Train IoU: %.3f" % (
            epoch, train_loss, train_iou))
        return train_loss, train_iou

    def _validate(self, val_data):
        self.model = self.model.eval()
        losses = []
        metrics = []
        for data_batch_ditct in data.DataLoader(val_data, batch_size=10, shuffle=False, drop_last=True):
            image = data_batch_ditct['image']
            mask = data_batch_ditct['mask']
            image = image.type(torch.FloatTensor)
            if CUDA:
                image = image.cuda()
            y_pred = self.model(Variable(image, volatile=True))
            if CUDA:
                loss = self.loss_fn(y_pred, mask.cuda())
            else:
                loss = self.loss_fn(y_pred, mask)
            losses.append(loss.data[0])

            preds = (y_pred.data > 0).float()

            if CUDA:
                metrics.append(iou_binary(preds, mask.to('cuda'), ignore=255, per_image=True))
            else:
                metrics.append(iou_binary(preds, mask, ignore=255, per_image=True))

        return np.mean(losses), np.mean(metrics)