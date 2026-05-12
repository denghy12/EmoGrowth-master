import argparse
import json

from utils.runtime import configure_visible_devices


def main():
    args = setup_parser().parse_args()
    # 配置文件就是一次实验的“说明书”。例如
    # exps/emotic_clif_formal_vit_b16_alpha_b5i3.json 里写明了数据集、
    # 增量协议、模型、训练轮数、loss 权重和结果保存路径。
    param = load_json(args.config)
    args = vars(args)
    args.update(param)

    # 先按配置里的 "device": ["0", "1"] 设置可见 GPU，再进入训练逻辑。
    configure_visible_devices(args)

    from trainer_ml import train

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        return json.load(data_file)


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Multi-label continual learning entrypoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/multi_label.json",
        help="Json file of settings.",
    )

    return parser


if __name__ == "__main__":
    main()
