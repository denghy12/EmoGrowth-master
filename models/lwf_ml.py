import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net_ml import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
# ### iScience ###
# init_epoch = 30
# init_lr = 0.0001
# init_weight_decay = 0.01
#
#
# epochs = 30
# lrate = 0.0001
# batch_size = 128
# weight_decay = 0.1
# num_workers = 8
# lamda = 3
### PNAS ###
init_epoch = 45
init_lr = 0.001
# init_weight_decay = 0.01
init_weight_decay = 0

epochs = 40
lrate = 0.001
batch_size = 128
# weight_decay = 0.01
weight_decay = 0
num_workers = 8
lamda = 3

class LwF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args)
        self.init_epoch = args.get("init_epochs", init_epoch)
        self.epochs = args.get("epochs", epochs)
        self.init_lr = args.get("init_lr", init_lr)
        self.lrate = args.get("lrate", lrate)
        self.init_weight_decay = args.get("init_weight_decay", init_weight_decay)
        self.weight_decay = args.get("weight_decay", weight_decay)
        self.batch_size = args.get("batch_size", batch_size)
        self.num_workers = args.get("num_workers", num_workers)
        self.lamda = args.get("lamda", lamda)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_x, train_y, train_dataset = data_manager.get_dataset(
            self._cur_task,
            source="train",
            ret_data=True,
        )
        self.cls_criterion = self._build_multilabel_criterion(
            train_y, target_mode="current"
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_dataset = data_manager.get_dataset(
            self._cur_task, source="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )
            self._init_train(train_loader, test_loader, optimizer)
        else:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=self.lrate,
                weight_decay=self.weight_decay,
            )
            self._update_representation(train_loader, test_loader, optimizer)

    def _init_train(self, train_loader, test_loader, optimizer):
        prog_bar = tqdm(range(self.init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = self.cls_criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map, train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader)
            test_map, test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Train_other_metrics {}, Test_other_metrics {}".format(
                self._cur_task,
                epoch + 1,
                self.init_epoch,
                losses / len(train_loader),
                train_map,
                test_map,
                train_other_metrics,
                test_other_metrics
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer):
        prog_bar = tqdm(range(self.epochs))
        kd_cost = torch.nn.MultiLabelSoftMarginLoss()
        trans = torch.nn.Sigmoid()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = self._targets_to_current_task(targets)
                loss_clf = self.cls_criterion(
                    logits[:, self._known_classes :], fake_targets
                )
                loss_kd = kd_cost(
                    logits[:, : self._known_classes],
                    trans(self._old_network(inputs)["logits"]),
                )
                loss = self.lamda * loss_kd + loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map, train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,train=True)
            test_map, test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Train_other_metrics {}, Test_other_metrics {}".format(
                self._cur_task,
                epoch + 1,
                self.epochs,
                losses / len(train_loader),
                train_map,
                test_map,
                train_other_metrics,
                test_other_metrics
            )
            prog_bar.set_description(info)
        logging.info(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
