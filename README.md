# Code for paper “EmoGrowth: Incremental Multi-label Emotion Decoding with Augmented Emotional Relation Graph". ICML 2025
> - authors: Kaicheng Fu, Changde Du, Jie Peng, Kunpeng Wang, Shuangchen Zhao, Xiaoyu Chen

We developed multi-label incremental learning code based on the PyCIL toolkit.

## Methods Reproduced

-  `FineTune`: Baseline method which simply updates parameters on new tasks.
-  `EWC`: Overcoming catastrophic forgetting in neural networks.
-  `LwF`: Learning without Forgetting.
-  `Experience Replay`: Baseline method with exemplar replay.
-  `Reservoir Replay`: A Single-Label Class Incremental Learning algorithm based on data replay, where the construction of a data buffer utilizes a reservoir sampling strategy.
-  `AGCN`: A Multi-Label Class Incremental Learning algorithm based on graph convolutional neural networks, where the graph adjacency matrix continuously expands as the tasks progress.
-  `PRS`: A Multi-Label Class Incremental Learning algorithm based on data replay, which improves upon the reservoir sampling strategy to ensure that the number of samples for each class in the data buffer is as balanced as possible.
-  `OCDM`: A Multi-Label Class Incremental Learning algorithm based on data replay that defines the construction and updating of the data buffer as an optimization problem to be solved.

Reference details are in Section B.2 of the paper.

Users can still extend other methods in PyCIL to multi-label scenarios.

## How To Use

### Dependencies

python==3.7.0
pytorch==1.8.1
torchvision==0.6.0
numpy==1.21.2
scipy==1.6.2
tqdm==4.62.3
pot==0.9.0

### Run experiment

#### Single-label

- `exps/cifar100_smoke.json` keeps the smoke setting and is now pinned to GPU `0,1`.
- `exps/cifar100_formal.json` expands the smoke setup into a formal CIFAR100 training config and is pinned to GPU `0,1`.

Run:

```bash
python main.py --config=./exps/cifar100_formal.json
```

#### Multi-label

1. Edit `exps/multi_label.json` for global settings.
2. Set `model_name` to one of `finetune`, `ewc`, `lwf`, `replay`, `agcn`, `clif`.
3. Set `data_root`, `aux_root`, and `results_root` in `exps/multi_label.json` if your data is not in the default location.
4. Run:

```bash
python main_ml.py --config=./exps/multi_label.json
```

`replay` supports `buffer_type` values `random`, `rs`, `ocdm`, and `prs`.

`trainer_ml.py` now defaults to a single formal training run. If you want the previous CLIF sensitivity sweep behaviour, set `sweep_hparams` to `true` in `exps/multi_label.json`.

### Hyper-parameters

When using PyCIL, you can edit the global parameters and algorithm-specific hyper-parameter in the corresponding json file.

These parameters include:

- **memory-size**: The total exemplar number in the incremental learning process. Assuming there are $K$ classes at the current stage, the model will preserve $\left[\frac{memory-size}{K}\right]$ exemplar per class.
- **init-cls**: The number of classes in the first incremental stage. Since there are different settings in CIL with a different number of classes in the first stage, our framework enables different choices to define the initial stage.
- **increment**: The number of classes in each incremental stage $i$, $i$ > 1. By default, the number of classes per incremental stage is equivalent per stage.
- **convnet-type**: The backbone network for the incremental model. According to the benchmark setting, `ResNet32` is utilized for `CIFAR100`, and `ResNet18` is used for `ImageNet`.
- **seed**: The random seed adopted for shuffling the class order. According to the benchmark setting, it is set to 1993 by default.

Other parameters in terms of model optimization can be modified in the corresponding Python file.
