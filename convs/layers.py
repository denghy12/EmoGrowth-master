import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[], batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.1,
                 with_output_nonlineartity=True):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        self.fcs = nn.ModuleList()
        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_features):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
        else:
            self.fcs.append(nn.Linear(in_features, out_features))
            if with_output_nonlineartity:
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(out_features, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

        self.reset_parameters()

    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                         nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()

    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input


class GINLayer(nn.Module):
    def __init__(self, mlp, eps=0.0, train_eps=True, residual=True):
        super(GINLayer, self).__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.residual = residual
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, input, adj):
        res = input

        # Aggregating neighborhood information
        neighs = torch.matmul(adj, res)

        # Reweighting the center node representation
        res = (1 + self.eps) * res + neighs

        # Updating node representations
        res = self.mlp(res)

        # Residual connection
        if self.residual:
            output = res + input
        else:
            output = res

        return output


class GIN(nn.Module):
    def __init__(self, num_layers, in_features, out_features, hidden_features=[],
                 eps=0.0, train_eps=True, residual=True, batchNorm=True,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(GIN, self).__init__()

        self.GINLayers = nn.ModuleList()

        if in_features != out_features:
            first_layer_res = False
        else:
            first_layer_res = True
        self.GINLayers.append(GINLayer(MLP(in_features, out_features, hidden_features, batchNorm,
                                           nonlinearity, negative_slope),
                                       eps, train_eps, first_layer_res))
        for i in range(num_layers - 1):
            self.GINLayers.append(GINLayer(MLP(out_features, out_features, hidden_features, batchNorm,
                                               nonlinearity, negative_slope),
                                           eps, train_eps, residual))

        self.reset_parameters()

    def reset_parameters(self):
        for l in self.GINLayers:
            l.reset_parameters()

    def forward(self, input, adj):
        for l in self.GINLayers:
            input = l(input, adj)
        return input


class FDModel(nn.Module):
    # 对应论文 framework.png 中间主干的“特征-标签融合”部分：
    # 1) x -> W_x^b -> z：把样本特征映射到隐藏空间；
    # 2) Emotion embeddings -> W_e^b -> alpha：把情绪标签嵌入变成每个类别的“重要性向量”；
    # 3) z 与 alpha 做 Hadamard Product：得到每个样本、每个情绪类别对应的 semantic-specific features。
    #
    # 注意：图里的 W_o^b 分类输出在本项目里不完全写在 FDModel 中，
    # FDModel 主要产出 semantic-specific features，后面的 IncrementalNet_CLIF.fc
    # 再把每个类别的特征变成最终 logits。
    def __init__(self, in_features_x, in_features_y, hidden_features, out_features,
                 in_layers1=1, out_layers=1, batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(FDModel, self).__init__()

        # NN1 对应图中的 W_x^b：把输入样本特征 x 映射成 z。
        # 在 EMOTIC ViT 配置中，x 是 768 维 ViT-B/16 bbox 特征，z 是 64 维。
        hidden_list = [512] * (in_layers1 - 1)
        self.NN1 = MLP(in_features_x, hidden_features, hidden_list,
                       batchNorm, nonlinearity, negative_slope)

        # NN2 对应图中的 W_e^b：把 GIN 得到的 emotion embeddings
        # 映射为 alpha，也就是每个情绪类别在 64 维特征上的门控/重要性向量。
        self.NN2 = nn.Linear(in_features_y, hidden_features)

        # NN3 是 Hadamard Product 之后的轻量变换，使融合后的类别相关特征
        # 仍保持 out_features 维，供后面的分类头逐类别输出 logits。
        hidden_list = [hidden_features] * (out_layers - 1)
        self.NN3 = MLP(hidden_features, out_features, hidden_list,
                       batchNorm, nonlinearity, negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        self.NN1.reset_parameters()
        nn.init.kaiming_uniform_(self.NN2.weight, nonlinearity='sigmoid')
        nn.init.constant_(self.NN2.bias, 0.0)
        self.NN3.reset_parameters()

    def forward(self, x, y):
        # x: 样本特征，shape=[batch_size, input_size]。
        # y: 情绪标签嵌入，shape=[当前已见类别数, 256]。
        x = self.NN1(x)  # b1 x h

        # feature 是样本级特征，用于后续 relation-based KD。
        # 它对应图中的 z，也就是经过 W_x^b 后的样本表示。
        feature = x

        # 把 emotion embeddings 转成 alpha 重要性向量。
        # sigmoid 让每个维度像一个 0-1 的软门控：这个情绪类别更关注哪些特征维度。
        y = torch.sigmoid(self.NN2(y))
        # 所有样本共享同一组类别嵌入。
        # x.unsqueeze(1): [B, 1, H]
        # y.unsqueeze(0): [1, C, H]
        # 相乘后 output: [B, C, H]，即每个样本对每个情绪类别都有一份专属特征。
        output = x.unsqueeze(1) * y.unsqueeze(0)  # b1 x b2 x h

        # 输出 semantic-specific features。后续分类头会把 [B, C, H]
        # 压成 [B, C]，得到每个情绪类别的预测分数。
        output = self.NN3(output)
        return feature,output
