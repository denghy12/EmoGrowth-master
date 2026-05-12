import torch
import torch.nn as nn
from convs.layers import GIN,FDModel
import math


# class MLP(torch.nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.feature_extract = torch.nn.Sequential(
#             torch.nn.Linear(2880, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 64),
#         )
#     def forward(self, x):
#         x = self.feature_extract(x)
#         return x

class CLIFNet_new(nn.Module):
    def __init__(self,args):
        super(CLIFNet_new, self).__init__()
        self.rand_seed = 0

        # 每个情绪类别对应一行初始 label embedding。
        # 后面会把它和 label_adj 一起送进 GIN，让类别嵌入带上情绪关系图的信息。
        # 架构图左侧/中间的 ERG 分支从这里开始：
        # 先给每个情绪类别准备一个初始 embedding，后面用 GIN-based encoder
        # 根据情绪关系图 label_adj 聚合邻居信息，得到 Emotion embeddings。
        n_class = args['total_class']
        self.label_embedding = nn.Parameter(torch.eye(n_class),
                                            requires_grad=False)

        # self.GIN_encoder = GIN(2, n_class, 256,
        #                        [math.ceil(256 / 2)])
        # self.FD_model = FDModel(in_features_x=args["input_size"], in_features_y=256, hidden_features=64, out_features=64, in_layers1=3,
        #                         out_layers=1, batchNorm=False, nonlinearity='relu', negative_slope=0.1)
        # 对应图中的 GIN-based encoder：
        # 输入是“类别节点初始向量 + 情绪关系图 A/label_adj”，
        # 输出是带有类别关系信息的 emotion embeddings。
        self.GIN_encoder = GIN(1, n_class, 256,
                               [math.ceil(256 / 2)])
        # FD_model 负责融合“样本特征”和“类别/标签嵌入”：
        # uni_features 是样本级特征，dis_features 是每个类别对应的判别特征。
        # 对应图中的 x -> W_x^b -> z、Emotion embeddings -> W_e^b -> alpha、
        # 以及 Hadamard Product 生成 semantic-specific features 的部分。
        self.FD_model = FDModel(in_features_x=args["input_size"], in_features_y=256, hidden_features=64, out_features=64, in_layers1=2,
                                out_layers=1, batchNorm=False, nonlinearity='relu', negative_slope=0.1)

        self.reset_parameters()

    def reset_parameters(self):
        # Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_embedding)
        self.GIN_encoder.reset_parameters()
        self.FD_model.reset_parameters()
        # self.cls_conv.reset_parameters()

    def get_config_optim(self):
        return [{'params': self.GIN_encoder.parameters()},
                {'params': self.FD_model.parameters()}]

    def forward(self,input,label_adj):
        # 第一步：用当前情绪关系图 label_adj 编码 label embedding。
        # 随着增量任务推进，label_adj 的大小也会从 5 类扩展到更多类别。
        # label_adj 是当前任务的情绪关系图 A^b；随着新类别加入，它会逐步变大。
        # GIN 输出的 label_embedding 对应图中的 Emotion embeddings。
        label_embedding = self.GIN_encoder(self.label_embedding[:label_adj.shape[0]], label_adj)
        # 第二步：把每个样本特征和每个类别嵌入融合，输出给上层分类器。
        # FDModel 输出：
        # feature: 样本级 z 特征，用于后面的 relation-based KD；
        # X: 每个类别一份的 semantic-specific features，后面送入 W_o^b 分类头。
        feature,X = self.FD_model(input, label_embedding)
        # output = self.cls_conv(X).squeeze(2)
        return{
            'uni_features':feature,
            'dis_features':X,
            'label_embedding':label_embedding
        }


