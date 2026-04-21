import copy
import logging
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.ml_losses import build_multilabel_loss
from utils.toolkit import tensor2numpy, accuracy
from utils.metrics import average_precision,all_metrics
try:
    from scipy.spatial.distance import cdist
except ImportError:
    def cdist(xa, xb, metric):
        if metric != "sqeuclidean":
            raise ValueError(f"Unsupported metric without scipy: {metric}")
        xa = np.asarray(xa)
        xb = np.asarray(xb)
        diff = xa[:, None, :] - xb[None, :, :]
        return np.sum(diff * diff, axis=2)
import os
EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._targets_memory_ml = []
        self.topk = 5
        self.total_sample = 0
        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.label_adj = None
        self.train_x_ld = None
        self.running_statistics = []
        self.loss_type = args.get("loss_type", "softmargin")
        self.loss_params = args.get("loss_params", {})
        self.kd_loss_type = args.get("kd_loss_type", "auto")
        self.kd_loss_params = args.get("kd_loss_params", {})
        self.train_assignment_mode = args.get("train_assignment_mode", "repeat_current")
        self.train_label_mode = args.get("train_label_mode", "current")
        self.agcnpp_protocol = args.get("agcnpp_protocol", "auto")
    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory_ml
        ), "Exemplar size error."
        return len(self._data_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def build_rehearsal_memory_ml(self, data_manager, per_class):
        self._construct_exemplar_unified_ml(data_manager, per_class)

    def build_rehearsal_memory_ml_rs(self, data_manager, memory_size):
        self._construct_exemplar_unified_ml_rs(data_manager, memory_size)

    def build_rehearsal_memory_ml_ocdm(self,data_manager,memory_size):
        self._construct_exemplar_unified_ml_ocdm(data_manager, memory_size)
    def build_rehearsal_memory_ml_prs(self,data_manager,memory_size):
        self._construct_exemplar_unified_ml_prs(data_manager, memory_size)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy

    def eval_multi_label_task(self, clif=False, agcn=False, return_outputs=False):
        return self._compute_multi_label_accuracy(
            self._network,
            self.test_loader,
            clif=clif,
            agcn=agcn,
            return_outputs=return_outputs,
        )

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _get_memory_ml(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory_ml)

    def _targets_are_seen_labels(self, targets):
        return targets.shape[1] == self._total_classes

    def _targets_to_current_task(self, targets):
        if self._known_classes == 0 or not self._targets_are_seen_labels(targets):
            return targets
        return targets[:, self._known_classes : self._total_classes]

    def _targets_to_seen(self, targets):
        if self._targets_are_seen_labels(targets) or self._known_classes == 0:
            return targets
        zeros = torch.zeros(
            [targets.shape[0], self._known_classes],
            dtype=targets.dtype,
            device=targets.device,
        )
        return torch.hstack((zeros, targets))

    def _resolve_train_label_mode(self):
        if self.train_label_mode in ("current", "seen"):
            return self.train_label_mode
        protocol = str(self.agcnpp_protocol).lower()
        if protocol == "cl":
            return "seen"
        return "current"

    def _validate_seen_labels_available(self, train_targets, target_mode):
        if target_mode != "seen" or self._known_classes == 0:
            return
        if not self._targets_are_seen_labels(train_targets):
            raise ValueError(
                "train_label_mode='seen' requires a label_session whose train labels "
                "cover all seen classes. Rebuild the EMOTIC split with "
                "--train-assignment-mode repeat_current --train-label-mode seen."
            )

    def _targets_for_train_label_mode(self, targets, target_mode):
        if target_mode == "seen":
            return self._targets_to_seen(targets)
        if target_mode == "current":
            return self._targets_to_current_task(targets)
        raise ValueError(f"Unsupported train label mode: {target_mode}")

    def _logits_for_train_label_mode(self, logits, target_mode):
        if target_mode == "seen":
            return logits[:, : self._total_classes]
        if target_mode == "current":
            return logits[:, self._known_classes : self._total_classes]
        raise ValueError(f"Unsupported train label mode: {target_mode}")

    def _target_row_to_global_indices(self, target_row):
        positive_indices = np.where(target_row == 1)[0]
        if target_row.shape[0] == self._total_classes:
            return positive_indices.tolist()
        return (positive_indices + self._known_classes).tolist()

    def _target_row_to_seen(self, target_row):
        if target_row.shape[0] == self._total_classes:
            return target_row
        return np.pad(target_row, (self._known_classes, 0), mode="constant")

    def _build_multilabel_criterion(self, train_targets, target_mode):
        if target_mode == "seen":
            criterion_targets = self._targets_to_seen(train_targets)
        elif target_mode == "current":
            criterion_targets = self._targets_to_current_task(train_targets)
        else:
            raise ValueError(f"Unsupported target mode: {target_mode}")
        return build_multilabel_loss(
            self.loss_type,
            criterion_targets,
            self._device,
            self.loss_params,
        )

    def _resolve_kd_loss_type(self):
        kd_loss_type = None if self.kd_loss_type is None else str(self.kd_loss_type).lower()
        if kd_loss_type in (None, "auto"):
            return "db" if str(self.loss_type).lower() == "db" else "softmargin"
        if kd_loss_type == "same":
            return str(self.loss_type).lower()
        return kd_loss_type

    def _build_kd_logits_criterion(self, soft_targets):
        kd_loss_type = self._resolve_kd_loss_type()
        if kd_loss_type == "softmargin":
            return nn.MultiLabelSoftMarginLoss()

        explicit_kd_loss_params = copy.deepcopy(self.kd_loss_params)
        kd_loss_params = copy.deepcopy(self.loss_params) if kd_loss_type == str(self.loss_type).lower() else {}
        kd_loss_params.update(explicit_kd_loss_params)
        if kd_loss_type == "db":
            kd_loss_params.setdefault("reweight_func", "rebalance")
            for key, value in {
                "focal": False,
                "neg_scale": 1.0,
                "init_bias": 0.0,
                "weight_norm": "by_batch",
            }.items():
                if key not in explicit_kd_loss_params:
                    kd_loss_params[key] = value

        return build_multilabel_loss(
            kd_loss_type,
            soft_targets,
            self._device,
            kd_loss_params,
        )

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _compute_multi_label_accuracy(
        self,
        model,
        loader,
        clif=False,
        train=False,
        update=False,
        er=False,
        agcn=False,
        return_outputs=False,
    ):
        model.eval()
        output = []
        label = []
        if clif or agcn:
            self._set_runtime_label_adj(model, self.label_adj.to(self._device))
        if clif == True:
            if update == False:
                for i, (inputs, targets) in enumerate(loader):
                    inputs = inputs.to(self._device)
                    label_adj = self.label_adj.to(self._device)
                    with torch.no_grad():
                        outputs,_ = model(inputs,label_adj)
                        if train:
                            outputs = outputs[:, self._known_classes:]
                            targets = self._targets_to_current_task(targets)
                    output.append(outputs.cpu().detach().numpy())
                    label.append(targets.cpu().detach().numpy())
            else:
                for i, (inputs, targets,_,_) in enumerate(loader):
                    inputs = inputs.to(self._device)
                    label_adj = self.label_adj.to(self._device)
                    with torch.no_grad():
                        outputs,_ = model(inputs,label_adj)
                        if train:
                            outputs = outputs[:, self._known_classes:]
                            targets = self._targets_to_current_task(targets)
                    output.append(outputs.cpu().detach().numpy())
                    label.append(targets.cpu().detach().numpy())
        else:
            if agcn == True:
                for i, (inputs, targets) in enumerate(loader):
                    inputs = inputs.to(self._device)
                    label_adj = self.label_adj.to(self._device)
                    with torch.no_grad():
                        outputs = model(inputs,label_adj)
                        if train:
                            outputs = outputs[:, self._known_classes:]
                            targets = self._targets_to_current_task(targets)
                    output.append(outputs.cpu().detach().numpy())
                    label.append(targets.cpu().detach().numpy())
            else:
                for i, (inputs, targets) in enumerate(loader):
                    inputs = inputs.to(self._device)
                    with torch.no_grad():
                        outputs = model(inputs)["logits"]
                        if train:
                            outputs = outputs[:, self._known_classes:]
                            targets = self._targets_to_current_task(targets)
                    output.append(outputs.cpu().detach().numpy())
                    label.append(targets.cpu().detach().numpy())
        output = np.concatenate(output)
        label = np.concatenate(label)
        _, map = average_precision(torch.from_numpy(output), torch.from_numpy(label))
        other_metrics = all_metrics(output,label)
        if return_outputs:
            return map, other_metrics, output, label
        return map,other_metrics

    def _set_runtime_label_adj(self, model, label_adj):
        module = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(module, "set_runtime_label_adj"):
            module.set_runtime_label_adj(label_adj)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

    def _construct_exemplar_unified_ml(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes), er method".format(m)
        )

        # Construct exemplars for new classes
        data, targets, data_set = data_manager.get_dataset(self._cur_task,source="train",ret_data=True)
        data = data.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        selected_index = []
        for fake_class_idx in range(data_manager.get_task_size(self._cur_task)):
            mask = np.where(targets[:,fake_class_idx] == 1)[0]
            if len(list(mask))<=m:
                random_mask = list(mask)
            else:
                random_mask = random.sample(list(mask),m)
            selected_index = selected_index + random_mask

        selected_index = list(set(selected_index))

        selected_exemplars = data[selected_index]
        self._data_memory = (
            np.concatenate((self._data_memory, selected_exemplars))
            if len(self._data_memory) != 0
            else selected_exemplars
        )
        for index in selected_index:
            label_list = self._target_row_to_global_indices(targets[index])
            self._targets_memory_ml.append(label_list)

    def _construct_exemplar_unified_ml_rs(self, data_manager, m):
        logging.info(
            "Constructing exemplars... ({} total memory), rs method".format(m)
        )

        # Construct exemplars for new classes
        data, targets, data_set = data_manager.get_dataset(self._cur_task,source="train",ret_data=True)
        data = data.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        for index in range(data.shape[0]):
            self.total_sample += 1
            if self.exemplar_size == m:
                if random.randint(1,self.total_sample)<=m:
                    chosen_one = random.randint(0,m-1)
                    self._data_memory[chosen_one] = data[index]
                    label_list = self._target_row_to_global_indices(targets[index])
                    self._targets_memory_ml[chosen_one] = label_list
            else:
                selected_exemplars = np.expand_dims(data[index],axis=0)
                self._data_memory = (
                    np.concatenate((self._data_memory, selected_exemplars))
                    if len(self._data_memory) != 0
                    else selected_exemplars
                )
                label_list = self._target_row_to_global_indices(targets[index])
                self._targets_memory_ml.append(label_list)

    def _construct_exemplar_unified_ml_prs(self, data_manager, m):
        logging.info(
            "Constructing exemplars... ({} total memory), prs method".format(m)
        )
        rou = 0
        ### reading data ###
        data, targets, data_set = data_manager.get_dataset(self._cur_task,source="train",ret_data=True)
        data = data.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        ### initialization and updating running statistics
        running_statistics = self.running_statistics
        statistic_temp = np.sum(targets,axis=0)
        if targets.shape[1] == self._total_classes:
            running_statistics = np.array(statistic_temp)
        else:
            running_statistics = running_statistics + statistic_temp.tolist()
            running_statistics = np.array(running_statistics)
        ### compute target partion P and M ###
        N = running_statistics
        P = np.power(N,rou)/np.sum(np.power(N,rou))
        M = P * m
        ### construct data buffer: self._data_memory and self._targets_memory ###
        for index in range(data.shape[0]):
            self.total_sample += 1
            if self.exemplar_size == m:        ### buffer in full ###
                ###sample-in###
                temp = self._target_row_to_seen(targets[index]) * np.exp(-N)
                W = temp / np.sum(temp)
                s = np.sum(M / N * W) ### probability of sample-in ###
                if random.random() < s:
                    ###sample-out###
                    ### compute the number of examples of each class in data memory ###
                    L = self._compute_memory_number_class(self._targets_memory_ml)
                    delta = L - P * np.sum(L)
                    probabilities = softmax(delta)
                    selected_cateogry = np.argmax(probabilities)
                    ### construct set Y ###
                    selected_index_Y = []
                    for index_m,temp in enumerate(self._targets_memory_ml):
                        if selected_cateogry in temp:
                            selected_index_Y.append(index_m)
                    ### construct set K from Y ###
                    q = np.zeros_like(delta)
                    q[delta<=0] = 1
                    n_star_set = []
                    for index_y in selected_index_Y:
                        n_star_set.append(self._compute_n_star(self._targets_memory_ml[index_y],q))
                    K_index = find_all_max_indexes(n_star_set)
                    selected_index_K = [selected_index_Y[key] for key in K_index]
                    ### Equation (5) ###
                    K_min = []
                    for index_k in selected_index_K:
                        Cki = self._compute_memory_number_class(self._targets_memory_ml[:index_k]+self._targets_memory_ml[index_k+1:])
                        K_min.append(np.sum(np.abs(Cki - P * np.sum(Cki))))
                    chosen_one = selected_index_K[np.argmin(np.array(K_min))]
                    self._data_memory[chosen_one] = data[index]
                    label_list = self._target_row_to_global_indices(targets[index])
                    self._targets_memory_ml[chosen_one] = label_list
            else:                              ### buffer is not full ###
                selected_exemplars = np.expand_dims(data[index],axis=0)
                self._data_memory = (
                    np.concatenate((self._data_memory, selected_exemplars))
                    if len(self._data_memory) != 0
                    else selected_exemplars
                )
                label_list = self._target_row_to_global_indices(targets[index])
                self._targets_memory_ml.append(label_list)

        self.running_statistics = running_statistics.tolist()

    def _compute_n_star(self,targets,q):
        multi_hot_vector_inv = np.ones(self._total_classes)
        multi_hot_vector_inv[targets] = 0
        star = np.dot(multi_hot_vector_inv,np.squeeze(q))
        return star

    def _compute_memory_number_class(self,targets):
        vector_targets = []
        for temp in targets:
            multi_hot_vector = np.zeros(self._total_classes)
            multi_hot_vector[temp] = 1
            vector_targets.append(list(multi_hot_vector))
        vector_targets = np.array(vector_targets)
        L = np.sum(vector_targets,axis=0)
        return L

    def _construct_exemplar_unified_ml_ocdm(self, data_manager, m):
        logging.info(
            "Constructing exemplars... ({} total memory), ocdm method".format(m)
        )
        data, targets, data_set = data_manager.get_dataset(self._cur_task, source="train", ret_data=True)
        data = data.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        if self.exemplar_size < m:
            r = m - self.exemplar_size
            chosen_num = min(data.shape[0],r)
            chosen_index = random.sample(list(range(data.shape[0])),chosen_num)
            selected_exemplars = data[chosen_index]
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            for index in chosen_index:
                label_list = self._target_row_to_global_indices(targets[index])
                self._targets_memory_ml.append(label_list)
            data = np.delete(data,chosen_index,axis=0)
            targets = np.delete(targets,chosen_index,axis=0)
        if data.shape[0] > 0:
            bt = data.shape[0]
            data = np.concatenate((self._data_memory, data))
            label_all = []
            for index in range(targets.shape[0]):
                label_list = self._target_row_to_global_indices(targets[index])
                label_all.append(label_list)
            targets = self._targets_memory_ml + label_all
            ### data targets M+b_t chose M ###
            p_target = np.ones(self._total_classes)/self._total_classes
            # ### orignial ###
            # for k in range(bt): ###共bt轮###
            #     distance_set = []
            #     print('k=',k)
            #     for j in range(m+bt-k):
            #         ### 删除一个然后计算距离 ###
            #         p_omega = self._compute_memory_distribution(targets[:j] + targets[j+1:])
            #         distance_set.append(self._compute_distance(p_omega,p_target))
            #     ### 选定删除的元素 ###
            #     min_index = np.argmin(np.array(distance_set))
            #     data = np.delete(data,min_index,axis=0)
            #     targets.pop(min_index)
            # self._data_memory = data
            # self._targets_memory_ml = targets
            ### another choice ###
            distance_set = []
            index_set = []
            for test in range(10000):
                random_index = random.sample(list(range(data.shape[0])), m)
                index_set.append(random_index)
                p_omega = self._compute_memory_distribution([targets[t] for t in random_index])
                distance_set.append(self._compute_distance(p_omega, p_target))
            min_index = np.argmin(np.array(distance_set))
            final_index = index_set[min_index]
            self._data_memory = data[final_index]
            self._targets_memory_ml = [targets[t] for t in final_index]

    def _compute_memory_distribution(self,targets):
        vector_targets = []
        for temp in targets:
            multi_hot_vector = np.zeros(self._total_classes)
            multi_hot_vector[temp] = 1
            vector_targets.append(list(multi_hot_vector))
        vector_targets = np.array(vector_targets)
        m = np.sum(vector_targets,axis=0)
        p_omega = m/np.sum(m)
        return p_omega
    def _compute_distance(self,p,q):
        kl = np.sum(np.where(p != 0, p * np.log(p / q), 0))
        return kl










    def sym_conditional_prob(self,y):
        adj = torch.matmul(y.t(), y)
        y_sum = torch.sum(y.t(), dim=1, keepdim=True)
        y_sum[y_sum < 1e-6] = 1e-6
        adj = adj / y_sum
        for i in range(adj.size(0)):
            adj[i, i] = 0.0
        adj = (adj + adj.t()) * 0.5
        return adj

    def sym_conditional_prob_update(self,soft_label,label_adj_old,y,ld=False):
        trans = torch.nn.Sigmoid()
        soft_label = trans(soft_label)
        if ld:
            soft_label = self.label_disambiguation(soft_label)
        adj = torch.zeros(self._total_classes,self._total_classes)
        adj[:self._known_classes,:self._known_classes] = label_adj_old
        adj[self._known_classes:self._total_classes,self._known_classes:self._total_classes] = self.sym_conditional_prob(y)
        ### upper right ###
        for i in range(self._known_classes):
            for j in range(self._known_classes,self._total_classes):
                adj[i,j] = torch.dot(soft_label[:,i].squeeze(),y[:,j-self._known_classes].squeeze())/torch.sum(y[:,j-self._known_classes])
        ### end ###
        ### lower left with Bayes ###
        for i in range(self._known_classes,self._total_classes):
            for j in range(self._known_classes):
                adj[i,j] = (adj[j,i] * torch.sum(y[:,i-self._known_classes])) / torch.sum(soft_label[:,j])
        ### end ###
        adj = (adj + adj.t()) * 0.5
        return adj,soft_label
    def label_disambiguation(self,soft_label,hard_label=False):
        T = 30
        alpha = 0.75
        sigma = 1.0
        m = self.train_x_ld.shape[0]
        q = soft_label.shape[1]
        gamma = 0.1
        ### compute similarity matrix S ###
        normalized_tensor = torch.nn.functional.normalize(self.train_x_ld, p=2, dim=1)

        S1 = torch.pdist(normalized_tensor,p=2)
        S = torch.zeros(m,m)
        index = 0
        for i in range(m):
            for j in range(i+1,m):
                S[i,j] = S1[index]
                S[j,i] = S1[index]
                index += 1
        S = torch.exp(-S**2/(2*sigma**2))

        ### compute propagation matrix H ###
        D = torch.diag(torch.sum(S, dim=0))
        D_inv = torch.inverse(D)
        H = torch.mm(S,D_inv)
        ### compute label confidence matrix F0 ###
        # F0 = torch.zeros(m,q)
        # for i in range(m):
        #     for j in range(q):
        #         F0[i,j] = soft_label[i,j]/torch.sum(soft_label[i])
        F0 = soft_label
        ### label propagation ###
        F= F0
        F_next = torch.zeros(m,q)
        for t in range(T):
            F_next = alpha * torch.mm(H.t(),F) + (1-alpha) * F0
            if torch.norm(F_next-F) < 1e-5:
                break
            else:
                F = F_next
        F_star = F_next
        ### final labeling confidence matrix F_hat ###
        # F_hat = torch.zeros(m,q)
        # for i in range(m):
        #     for j in range(q):
        #         F_hat[i,j] = F_star[i,j]/torch.sum(F_star[i])
        # print('F_hat=',F_hat.numpy())
        F_hat = F_star
        ### candidate label set ###
        if hard_label:
            hard_label = torch.zeros(m,q)
            for i in range(m):
                for j in range(q):
                    if F_hat[i,j] > gamma:
                        hard_label[i,j] = 1
            return hard_label
        else:
            return F_hat

def softmax(x):
    e_x = np.exp(x - np.max(x)) # 减去最大值，避免指数溢出
    return e_x / np.sum(e_x)

def find_all_max_indexes(lst):
    max_value = max(lst)
    max_indexes = [index for index, value in enumerate(lst) if value == max_value]
    return max_indexes
