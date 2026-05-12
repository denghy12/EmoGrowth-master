import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from utils.inc_net_ml import IncrementalNet_CLIF
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.metrics import *
from utils.loss import LinkPredictionLoss_cosine
from torch.utils.data import Dataset,TensorDataset
### iscience ###

# # init_epoch = 20
# init_epoch = 50
# init_lr = 0.0001
# init_weight_decay = 0.005
#
#
# # epochs = 10
# lrate = 0.0001
# batch_size = 128
# weight_decay = 0.005
# num_workers = 8
# lamda_kd_logits = 1


## iscience ###
init_epoch = 40
init_lr = 0.0001
init_weight_decay = 0.005


epochs = 30
lrate = 0.0001
batch_size = 128
weight_decay = 0.005
num_workers = 8
lamda_kd_logits = 1

# ### pnas ###
# init_epoch = 50
# init_lr = 0.001
# # init_weight_decay = 0.01
# init_weight_decay = 0
#
# epochs = 40
# lrate = 0.001
# batch_size = 128
# # weight_decay = 0.01
# weight_decay = 0
# num_workers = 8
# lamda_kd_logits = 1


# lamda_le = 0.005
# lamda_kd_relation = 1

class CLIF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet_CLIF(args)
        # 下面这些训练参数都来自 exps/emotic_clif_formal_vit_b16_alpha_b5i3.json：
        # 初始任务训练 40 轮，后续增量任务训练 30 轮，batch_size=128，优化器用 Adam。
        self.init_epoch = args.get("init_epochs", init_epoch)
        self.epochs = args.get("epochs", epochs)
        self.init_lr = args.get("init_lr", init_lr)
        self.lrate = args.get("lrate", lrate)
        self.init_weight_decay = args.get("init_weight_decay", init_weight_decay)
        self.weight_decay = args.get("weight_decay", weight_decay)
        self.batch_size = args.get("batch_size", batch_size)
        self.num_workers = args.get("num_workers", num_workers)
        self.lamda_kd_logits = args.get("lamda_kd_logits", lamda_kd_logits)
        self.ld = args["ld"]
        # 三个关键权重：
        # lamda_le 控制标签嵌入是否贴合情绪关系图；
        # lamda_kd_relation_data 控制旧模型样本关系蒸馏；
        # lamda_kd_relation_aff 控制情感维度关系蒸馏。
        self.lamda_kd_relation_aff = args['lamda_kd_relation_aff']
        self.lamda_kd_relation_data = args['lamda_kd_relation_data']
        self.lamda_le = args['lamda_le']
        self.subject = args["subject"]
        self.dataset = args.get("dataset", "")

    @staticmethod
    def _normalize_label_embedding(label_embedding, label_adj):
        expected = label_adj.shape[0]
        if label_embedding.shape[0] == expected:
            return label_embedding
        # 多 GPU DataParallel 会把每张卡上的 label embedding 拼起来；
        # 但每张卡算的是同一份类别嵌入，所以取前 expected 行即可。
        return label_embedding[:expected]

    def after_task(self):
        # 一个任务结束后，把当前模型冻结成 old_network。
        # 下一轮新增类别时，它负责提供旧类别的软标签和旧特征关系。
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        self._old_label_adj = self.label_adj

    def incremental_train(self, data_manager):
        self._cur_task += 1
        # 推进当前已见类别范围。B5I3 下：
        # task0 学 0-4 类，task1 扩到 0-7 类，task2 扩到 0-10 类，以此类推。
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_x,train_y,train_affective_dimension,train_dataset = data_manager.get_dataset(
            self._cur_task,
            source="train",
            ret_data=True,
            affective=True
        )
        self._active_train_label_mode = self._resolve_train_label_mode()
        self._validate_seen_labels_available(train_y, self._active_train_label_mode)
        self.cls_criterion = self._build_multilabel_criterion(
            train_y, target_mode=self._active_train_label_mode
        )
        self.train_x_ld = train_x

        # 构建或更新情绪关系图 label_adj。
        # 第 0 个任务只有真实标签；后续任务会用 old_network 的预测补旧类别信息。
        if self._cur_task > 0:
            with torch.no_grad():
                soft_label_known,_ = self._old_network(train_x.to(self._device), self.label_adj.to(self._device))
            self.kd_logits_criterion = self._build_kd_logits_criterion(torch.sigmoid(soft_label_known).cpu())
            if self._active_train_label_mode == "seen":
                self.label_adj = self.sym_conditional_prob(train_y)
                self.soft_label = torch.sigmoid(soft_label_known.cpu())
            else:
                current_train_y = self._targets_to_current_task(train_y)
                self.label_adj,self.soft_label = self.sym_conditional_prob_update(soft_label_known.cpu(), self.label_adj, current_train_y, ld=self.ld)
            # 后续任务训练时需要四类信息：
            # 输入特征、multi-hot 标签、old model 对旧类别的 logits、情感维度向量。
            # self.train_loader = DataLoader(
            #     TensorDataset(train_x,train_y,self.soft_label), batch_size=batch_size, shuffle=True, num_workers=num_workers
            # )
            ### 引入消歧计算标签共生但不用于计算知识蒸馏 args_ld is true###
            self.train_loader = DataLoader(
                TensorDataset(train_x,train_y,soft_label_known.cpu(),train_affective_dimension),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            ######
        else:
            # 第一个任务没有旧模型，直接用当前任务真实标签统计情绪共现关系图。
            label_targets = self._targets_for_train_label_mode(train_y, self._active_train_label_mode)
            self.label_adj = self.sym_conditional_prob(label_targets)
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
        self._set_runtime_label_adj(self._network, self.label_adj.to(self._device))
        # 第一个任务还没有旧模型，所以只训练“分类 + 标签图结构”；
        # 从第二个任务开始，才加入知识蒸馏来缓解灾难性遗忘。
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
            )  # 1e-5
            self._update_representation(train_loader, test_loader, optimizer)

    def _init_train(self, train_loader, test_loader, optimizer):
        prog_bar = tqdm(range(self.init_epoch))
        emb_cost = LinkPredictionLoss_cosine()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            loss_sums = {
                "loss_total": 0.0,
                "loss_clf": 0.0,
                "loss_le": 0.0,
                "weighted_loss_le": 0.0,
            }
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                label_adj = self.label_adj.to(self._device)
                loss_adj = label_adj + torch.eye(label_adj.data.size(0), dtype=label_adj.data.dtype,device=label_adj.data.device) ###identity matrix included###
                logits,label_embedding = self._network(inputs,label_adj)
                label_embedding = self._normalize_label_embedding(label_embedding, label_adj)
                loss_targets = self._targets_for_train_label_mode(targets, self._active_train_label_mode)
                loss_logits = self._logits_for_train_label_mode(logits, self._active_train_label_mode)
                loss_clf = self.cls_criterion(loss_logits, loss_targets)
                # loss_le 要求 label_embedding 重构 label_adj，也就是让类别嵌入保留情绪关系图结构。
                loss_le = emb_cost(label_embedding,loss_adj)
                loss = loss_clf + self.lamda_le * loss_le
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                loss_sums["loss_total"] += self._loss_value(loss)
                loss_sums["loss_clf"] += self._loss_value(loss_clf)
                loss_sums["loss_le"] += self._loss_value(loss_le)
                loss_sums["weighted_loss_le"] += self._loss_value(self.lamda_le * loss_le)
            self._append_loss_details(epoch + 1, "init", loss_sums, len(train_loader))
            train_map,train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,clif=True)
            test_map,test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader,clif=True)

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
        emb_cost = LinkPredictionLoss_cosine()
        trans = torch.nn.Sigmoid()
        self._set_runtime_label_adj(self._network, self.label_adj.to(self._device))
        self._set_runtime_label_adj(self._old_network, self._old_label_adj.to(self._device))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            loss_sums = {
                "loss_total": 0.0,
                "loss_clf": 0.0,
                "loss_kd_logits": 0.0,
                "loss_clf_plus_kd_logits": 0.0,
                "loss_le": 0.0,
                "loss_kd_model": 0.0,
                "loss_kd_aff": 0.0,
                "weighted_loss_kd_logits": 0.0,
                "weighted_loss_le": 0.0,
                "weighted_loss_kd_model": 0.0,
                "weighted_loss_kd_aff": 0.0,
            }
            for i, (inputs, targets,soft_targets,affective_dimension) in enumerate(train_loader):
                inputs, targets,soft_targets,affective_dimension = inputs.to(self._device), targets.to(self._device), soft_targets.to(self._device),affective_dimension.to(self._device)
                ### 引入消歧计算标签共生但不用于计算知识蒸馏 args_ld is true###
                # old model 输出的 logits 转成概率，作为旧类别的软标签蒸馏目标。
                soft_targets = trans(soft_targets)
                ######

                label_adj = self.label_adj.to(self._device)
                loss_adj = label_adj + torch.eye(label_adj.data.size(0), dtype=label_adj.data.dtype,device=label_adj.data.device) ###identity matrix included###
                logits,label_embedding = self._network(inputs,label_adj)
                label_embedding = self._normalize_label_embedding(label_embedding, label_adj)

                ### 基于样本相似度的知识蒸馏 ###
                # 关系蒸馏不是逐个样本对齐，而是对齐一个 batch 内样本两两相似度：
                # old feature -> new feature 用来保留旧模型知识；
                # affective dimension -> new feature 用来注入情感语义结构。
                _, _, feature_old = self._old_network(inputs,self._old_label_adj.to(self._device),kd=True)
                logits_1, label_embedding_1, feature_new = self._network(inputs,label_adj,kd=True)
                ######

                loss_targets = self._targets_for_train_label_mode(targets, self._active_train_label_mode)
                loss_logits = self._logits_for_train_label_mode(logits, self._active_train_label_mode)
                loss_clf = self.cls_criterion(
                    loss_logits, loss_targets
                )

                loss_le = emb_cost(label_embedding,loss_adj)
                loss_kd_logits = self.kd_logits_criterion(logits[:, : self._known_classes], soft_targets)
                loss_kd_relation_1 = self.compute_relation_based_kd_loss(feature_old,feature_new)
                loss_kd_relation_2 = self.compute_relation_based_kd_loss(affective_dimension,feature_new)
                loss_clf_plus_kd_logits = loss_clf + self.lamda_kd_logits * loss_kd_logits
                # loss = loss_clf + self.lamda_le * loss_le + self.lamda_kd_relation_data * loss_kd_relation_1 + lamda_kd_logits * loss_kd_logits ### 删除情感维度蒸馏 ###
                # loss = loss_clf + self.lamda_le * loss_le + self.lamda_kd_relation_aff * loss_kd_relation_2 + lamda_kd_logits * loss_kd_logits ### 删除样本蒸馏 ###
                # loss = loss_clf + self.lamda_le * loss_le + lamda_kd_logits * loss_kd_logits ### 删除relation蒸馏 ###
                # 后续任务的完整 CLIF 目标：
                # 分类损失 + 标签图损失 + 旧 logits 蒸馏 + 旧模型关系蒸馏 + 情感维度关系蒸馏。
                loss = loss_clf + self.lamda_le * loss_le + self.lamda_kd_relation_data * loss_kd_relation_1 + self.lamda_kd_relation_aff * loss_kd_relation_2 + self.lamda_kd_logits * loss_kd_logits
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                loss_sums["loss_total"] += self._loss_value(loss)
                loss_sums["loss_clf"] += self._loss_value(loss_clf)
                loss_sums["loss_kd_logits"] += self._loss_value(loss_kd_logits)
                loss_sums["loss_clf_plus_kd_logits"] += self._loss_value(loss_clf_plus_kd_logits)
                loss_sums["loss_le"] += self._loss_value(loss_le)
                loss_sums["loss_kd_model"] += self._loss_value(loss_kd_relation_1)
                loss_sums["loss_kd_aff"] += self._loss_value(loss_kd_relation_2)
                loss_sums["weighted_loss_kd_logits"] += self._loss_value(
                    self.lamda_kd_logits * loss_kd_logits
                )
                loss_sums["weighted_loss_le"] += self._loss_value(self.lamda_le * loss_le)
                loss_sums["weighted_loss_kd_model"] += self._loss_value(
                    self.lamda_kd_relation_data * loss_kd_relation_1
                )
                loss_sums["weighted_loss_kd_aff"] += self._loss_value(
                    self.lamda_kd_relation_aff * loss_kd_relation_2
                )
            self._append_loss_details(epoch + 1, "update", loss_sums, len(train_loader))
            train_map, train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,clif=True,update=True,train=True)
            test_map, test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader,clif=True)
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

    def fake_target_gen(self,targets):
        return torch.hstack((torch.zeros([targets.shape[0],self._known_classes]).to(self._device),targets))

    def compute_relation_based_kd_loss(self,feature_old,feature_new):
        # 把一个 batch 内样本两两的余弦相似度看成“关系矩阵”。
        # 这个 loss 惩罚旧关系矩阵和新关系矩阵的差异，
        # 所以模型不是只记住单个样本输出，还要保留样本之间的相对结构。
        remove = 0
        feature_old = feature_old-torch.mean(feature_old,dim=0)
        feature_new = feature_new-torch.mean(feature_new,dim=0)
        RSM_old = torch.nn.functional.cosine_similarity(feature_old.unsqueeze(1), feature_old.unsqueeze(0), dim=-1)
        RSM_new = torch.nn.functional.cosine_similarity(feature_new.unsqueeze(1), feature_new.unsqueeze(0), dim=-1)
        if self.subject == 'visual' or str(self.dataset).upper() == "EMOTIC":
            loss = (RSM_old - torch.diag_embed(torch.diag(RSM_old))) - (RSM_new - torch.diag_embed(torch.diag(RSM_new)))
        else:
            loss = torch.atanh(RSM_old-torch.diag_embed(torch.diag(RSM_old)))-torch.atanh(RSM_new-torch.diag_embed(torch.diag(RSM_new)))
        if len(torch.where(torch.isinf(loss))[0])>0:
            # print('inf appears.')
            remove = len(torch.where(torch.isinf(loss))[0])
            # print('remove=',remove)
            # print('position',torch.where(torch.isinf(loss)))
            cnn_f = RSM_old-torch.diag_embed(torch.diag(RSM_old))
            rsa = RSM_new-torch.diag_embed(torch.diag(RSM_new))
            # print(cnn_f[torch.where(torch.isinf(loss))])
            # print(rsa[torch.where(torch.isinf(loss))])
            loss = torch.where(torch.isinf(loss), torch.full_like(loss, 0), loss)

        if len(torch.where(torch.isnan(loss))[0])>0:
            # print('nan appears.')
            remove = len(torch.where(torch.isnan(loss))[0])
            # print('remove=',remove)
            # print('position',torch.where(torch.isnan(loss)))
            cnn_f = RSM_old-torch.diag_embed(torch.diag(RSM_old))
            rsa = RSM_new-torch.diag_embed(torch.diag(RSM_new))
            # print(cnn_f[torch.where(torch.isnan(loss))])
            # print(rsa[torch.where(torch.isnan(loss))])
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
        loss = torch.pow(loss, 2)
        num = (feature_old.size(0)*(feature_old.size(0)-1))/2
        return loss.sum()/2/(num-remove//2)
