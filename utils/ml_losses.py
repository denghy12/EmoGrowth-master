import math

import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-12


def _to_device_tensor(values, device):
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32)
    return torch.tensor(values, device=device, dtype=torch.float32)


def compute_class_stats(targets, device):
    targets = _to_device_tensor(targets, device)
    if targets.ndim != 2:
        raise ValueError("Multi-label targets must be a 2D tensor.")
    num_samples = max(int(targets.shape[0]), 1)
    class_freq = targets.sum(dim=0)
    neg_class_freq = num_samples - class_freq
    return {
        "targets": targets,
        "num_samples": num_samples,
        "class_freq": class_freq,
        "neg_class_freq": neg_class_freq,
    }


def build_pos_weight(class_freq, neg_class_freq):
    class_freq = class_freq.clamp(min=1.0)
    neg_class_freq = neg_class_freq.clamp(min=1.0)
    return neg_class_freq / class_freq


def effective_num_weights(class_freq, beta=0.9999):
    class_freq = class_freq.clamp(min=1.0)
    beta_tensor = class_freq.new_tensor(beta)
    effective_num = 1.0 - torch.pow(beta_tensor, class_freq)
    weights = (1.0 - beta_tensor) / effective_num.clamp(min=EPS)
    weights = weights / weights.sum().clamp(min=EPS) * class_freq.numel()
    return weights


class BCEPosWeightLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        return self.criterion(logits, targets.float())


class ClassBalancedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, targets):
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = loss * self.class_weights.unsqueeze(0)
        return loss.mean()


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_neg=4.0,
        gamma_pos=1.0,
        clip=0.05,
        eps=1e-8,
        reduction="mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(pos_probs.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(neg_probs.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = pos_probs * targets + neg_probs * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)

        loss = -loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class DistributionBalancedLoss(nn.Module):
    def __init__(
        self,
        class_freq,
        neg_class_freq,
        num_samples,
        loss_weight=1.0,
        focal=True,
        balance_param=2.0,
        gamma=2.0,
        reweight_func="rebalance",
        weight_norm=None,
        map_alpha=0.1,
        map_beta=10.0,
        map_gamma=0.3,
        neg_scale=5.0,
        init_bias=0.05,
        cb_beta=0.9,
        cb_mode="average_w",
        reduction="mean",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.focal = focal
        self.gamma = gamma
        self.balance_param = balance_param
        self.reweight_func = reweight_func
        self.weight_norm = weight_norm
        self.map_alpha = map_alpha
        self.map_beta = map_beta
        self.map_gamma = map_gamma
        self.neg_scale = neg_scale
        self.init_bias_scale = init_bias
        self.cb_beta = cb_beta
        self.cb_mode = cb_mode
        self.reduction = reduction

        class_freq = class_freq.clamp(min=EPS)
        neg_class_freq = neg_class_freq.clamp(min=EPS)
        num_samples_tensor = class_freq.new_tensor(float(num_samples))
        safe_class_freq = class_freq.clamp(max=max(float(num_samples) - EPS, EPS))

        self.register_buffer("class_freq", class_freq)
        self.register_buffer("neg_class_freq", neg_class_freq)
        self.register_buffer("train_num", num_samples_tensor)
        self.register_buffer("freq_inv", 1.0 / class_freq)
        self.register_buffer("propotion_inv", num_samples_tensor / class_freq)

        bias_ratio = (num_samples_tensor / safe_class_freq - 1.0).clamp(min=EPS)
        init_bias_tensor = -torch.log(bias_ratio) * self.init_bias_scale / max(self.neg_scale, EPS)
        init_bias_tensor = init_bias_tensor.clamp(min=-20.0, max=20.0)
        self.register_buffer("init_bias", init_bias_tensor)

    def forward(self, logits, targets):
        targets = targets.float().clamp(min=0.0, max=1.0)
        weight = self._reweight(targets)
        logits, weight = self._apply_logit_regularization(targets, logits, weight)

        if self.focal:
            base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            pt = torch.exp(-base_loss)
            loss = base_loss * weight
            loss = self.balance_param * torch.pow(1.0 - pt, self.gamma) * loss
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = loss * weight

        loss = loss * self.loss_weight
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

    def _reweight(self, targets):
        if self.reweight_func is None:
            weight = torch.ones_like(targets)
        elif self.reweight_func == "rebalance":
            repeat_rate = torch.sum(targets * self.freq_inv.unsqueeze(0), dim=1, keepdim=True).clamp(min=EPS)
            pos_weight = self.freq_inv.unsqueeze(0) / repeat_rate
            weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        elif self.reweight_func == "inv":
            weight = self.propotion_inv.unsqueeze(0).expand_as(targets)
        elif self.reweight_func == "sqrt_inv":
            weight = torch.sqrt(self.propotion_inv).unsqueeze(0).expand_as(targets)
        elif self.reweight_func == "cb":
            weight = self._class_balanced_weight(targets)
        else:
            raise ValueError(f"Unsupported DB reweight_func: {self.reweight_func}")

        if self.weight_norm == "by_instance":
            max_by_instance = weight.max(dim=1, keepdim=True).values.clamp(min=EPS)
            weight = weight / max_by_instance
        elif self.weight_norm == "by_batch":
            weight = weight / weight.max().clamp(min=EPS)

        return weight

    def _class_balanced_weight(self, targets):
        beta = self.class_freq.new_tensor(self.cb_beta)
        if self.cb_mode == "by_class":
            weight = (1.0 - beta) / (1.0 - torch.pow(beta, self.class_freq)).clamp(min=EPS)
            return weight.unsqueeze(0).expand_as(targets)

        if self.cb_mode == "average_n":
            avg_n = torch.sum(targets * self.class_freq.unsqueeze(0), dim=1, keepdim=True)
            avg_n = avg_n / torch.sum(targets, dim=1, keepdim=True).clamp(min=EPS)
            return (1.0 - beta) / (1.0 - torch.pow(beta, avg_n)).clamp(min=EPS)

        if self.cb_mode == "average_w":
            class_weight = (1.0 - beta) / (1.0 - torch.pow(beta, self.class_freq)).clamp(min=EPS)
            weight = torch.sum(targets * class_weight.unsqueeze(0), dim=1, keepdim=True)
            weight = weight / torch.sum(targets, dim=1, keepdim=True).clamp(min=EPS)
            return weight

        if self.cb_mode == "min_n":
            large = targets.new_full(targets.shape, 100000.0)
            positive_freq = targets * self.class_freq.unsqueeze(0) + (1.0 - targets) * large
            min_n = positive_freq.min(dim=1, keepdim=True).values
            return (1.0 - beta) / (1.0 - torch.pow(beta, min_n)).clamp(min=EPS)

        raise ValueError(f"Unsupported DB cb_mode: {self.cb_mode}")

    def _apply_logit_regularization(self, targets, logits, weight):
        logits = logits + self.init_bias.unsqueeze(0)
        if weight is None:
            weight = torch.ones_like(targets)
        logits = logits * ((1.0 - targets) * self.neg_scale + targets)
        weight = weight / self.neg_scale * (1.0 - targets) + weight * targets
        return logits, weight


def build_multilabel_loss(loss_type, targets, device, loss_params=None):
    loss_type = (loss_type or "softmargin").lower()
    loss_params = loss_params or {}
    reweight_func = loss_params.get("reweight_func", "rebalance")
    if isinstance(reweight_func, str):
        reweight_func = reweight_func.lower()
    stats = compute_class_stats(targets, device)
    class_freq = stats["class_freq"]
    neg_class_freq = stats["neg_class_freq"]
    num_samples = stats["num_samples"]

    if loss_type == "softmargin":
        return nn.MultiLabelSoftMarginLoss()

    if loss_type == "bce_pos_weight":
        pos_weight = build_pos_weight(class_freq, neg_class_freq)
        return BCEPosWeightLoss(pos_weight)

    if loss_type == "cb_bce":
        beta = loss_params.get("beta", 0.9999)
        class_weights = effective_num_weights(class_freq, beta=beta)
        return ClassBalancedBCELoss(class_weights)

    if loss_type == "asl":
        return AsymmetricLossMultiLabel(
            gamma_neg=loss_params.get("gamma_neg", 4.0),
            gamma_pos=loss_params.get("gamma_pos", 1.0),
            clip=loss_params.get("clip", 0.05),
            eps=loss_params.get("eps", 1e-8),
            reduction=loss_params.get("reduction", "mean"),
        )

    if loss_type == "db":
        return DistributionBalancedLoss(
            class_freq=class_freq,
            neg_class_freq=neg_class_freq,
            num_samples=num_samples,
            loss_weight=loss_params.get("loss_weight", 1.0),
            focal=loss_params.get("focal", True),
            balance_param=loss_params.get("balance_param", 2.0),
            gamma=loss_params.get("gamma", 2.0),
            reweight_func=reweight_func,
            weight_norm=loss_params.get("weight_norm"),
            map_alpha=loss_params.get("map_alpha", 0.1),
            map_beta=loss_params.get("map_beta", 10.0),
            map_gamma=loss_params.get("map_gamma", 0.3),
            neg_scale=loss_params.get("neg_scale", 5.0),
            init_bias=loss_params.get("init_bias", 0.05),
            cb_beta=loss_params.get("cb_beta", 0.9),
            cb_mode=loss_params.get("cb_mode", "average_w"),
            reduction=loss_params.get("reduction", "mean"),
        )

    raise ValueError(f"Unsupported multi-label loss_type: {loss_type}")
