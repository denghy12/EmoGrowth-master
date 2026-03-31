import argparse
import json

from utils.runtime import configure_visible_devices


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)

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
