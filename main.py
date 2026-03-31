import argparse
import json

from utils.runtime import configure_visible_devices


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    cli_args = vars(args)
    cli_args.update(param)

    configure_visible_devices(cli_args)

    from trainer import train

    train(cli_args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        return json.load(data_file)


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Single-label continual learning entrypoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/finetune.json",
        help="Json file of settings.",
    )
    return parser


if __name__ == "__main__":
    main()
