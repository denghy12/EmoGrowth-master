import copy
import logging
import torch
from torch import nn
from convs.mlp import MLP
from convs.linears import SimpleLinear
from convs.CLIFModel import CLIFNet_new
from convs.AGCNModel import AGCNNet,gcn

def get_mlpnet(args):
    name = args["convnet_type"].lower()
    if name == "mlp":
        return MLP(args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))

def get_clifnet(args):
    name = args["convnet_type"].lower()
    if name == "clif":
        return CLIFNet_new(args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))

def get_agcnnet(args):
    name = args["convnet_type"].lower()
    if name == "agcn":
        return AGCNNet(args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()

        self.mlpnet = get_mlpnet(args)
        self.fc = None
        self.feature_dim = args['feature_dim']

    def extract_vector(self, x):
        return self.mlpnet(x)["features"]

    def forward(self, x):
        x = self.mlpnet(x)
        out = self.fc(x["features"])
        """
        {
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class IncrementalNet(BaseNet):
    def __init__(self, args):
        super().__init__(args)
    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.mlpnet(x)
        out = self.fc(x["features"])
        out.update(x)
        return out


class BaseNet_CLIF(nn.Module):
    def __init__(self, args):
        super(BaseNet_CLIF, self).__init__()

        self.clifnet = get_clifnet(args)
        self.fc = None
        self.feature_dim = args['feature_dim']
        self.register_buffer("runtime_label_adj", torch.empty(0))

    def extract_vector(self, x):
        return self.mlpnet(x)["features"]

    def set_runtime_label_adj(self, label_adj):
        self.runtime_label_adj = label_adj

    def _resolve_label_adj(self, label_adj):
        if self.runtime_label_adj.numel() > 0:
            return self.runtime_label_adj
        return label_adj

    def forward(self, x,label_adj):
        label_adj = self._resolve_label_adj(label_adj)
        output = self.clifnet(x,label_adj)
        logits = self.fc(output["dis_features"]).squeeze(2)
        label_embedding = output['label_embedding']
        return logits,label_embedding

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class IncrementalNet_CLIF(BaseNet_CLIF):
    def __init__(self, args):
        super().__init__(args)
        self.feature_dim = args['feature_dim']
    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.bias.data.shape[0]
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        # 对应论文 framework.png 中的 W_o^b。
        # FDModel 输出的是 [B, C, H] 的 semantic-specific features；
        # 这里用 groups=C 的 Conv1d 给每个情绪类别各自分配一个分类器，
        # 把每个类别的 H 维特征压成 1 个 logit。
        fc = nn.Conv1d(out_dim,out_dim,in_dim,groups=out_dim)

        return fc

    def forward(self,x,label_adj,kd=False):
        label_adj = self._resolve_label_adj(label_adj)
        output = self.clifnet(x,label_adj)
        # output["dis_features"] 来自 FDModel，对应图中的 semantic-specific features。
        # 经过上面的分组分类器后得到 logits，shape=[batch_size, 当前已见类别数]。
        logits = self.fc(output["dis_features"]).squeeze(2)
        label_embedding = output['label_embedding']
        uni_features = output['uni_features']
        if kd:
            return logits,label_embedding,uni_features
        else:
            return logits,label_embedding

class BaseNet_AGCN(nn.Module):
    def __init__(self, args):
        super(BaseNet_AGCN, self).__init__()

        self.agcnnet = get_agcnnet(args)
        self.fc = None
        self.feature_dim = args['feature_dim']
        self.register_buffer("runtime_label_adj", torch.empty(0))

    def extract_vector(self, x):
        return self.mlpnet(x)["features"]

    def set_runtime_label_adj(self, label_adj):
        self.runtime_label_adj = label_adj

    def _resolve_label_adj(self, label_adj):
        if self.runtime_label_adj.numel() > 0:
            return self.runtime_label_adj
        return label_adj

    def forward(self, x,label_adj):
        pass

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class IncrementalNet_AGCN(BaseNet_AGCN):
    def __init__(self, args):
        super().__init__(args)
        self.feature_dim = args['feature_dim']
        self.gcn_layer = gcn(self.feature_dim,1)

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.bias.data.shape[0]
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = torch.nn.Linear(in_dim,out_dim)
        return fc

    def forward(self,x,label_adj):
        label_adj = self._resolve_label_adj(label_adj)
        x = self.agcnnet(x)['features']
        ya = self.fc(x)
        label_embedding = x.unsqueeze(2).repeat(1, 1, label_adj.shape[0]).permute(0, 2, 1)
        label_embedding = label_embedding * self.fc.weight
        yb = self.gcn_layer(label_embedding,label_adj)
        logits =  ya + yb
        return logits
