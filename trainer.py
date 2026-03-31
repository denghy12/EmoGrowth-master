import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = args["seed"] if isinstance(args["seed"], list) else [args["seed"]]
    device_cfg = list(args["device"])

    for seed in seed_list:
        run_args = dict(args)
        run_args["seed"] = seed
        run_args["device"] = list(device_cfg)
        _train(run_args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = f"logs/{args['model_name']}/{args['dataset']}/{init_cls}/{args['increment']}"
    os.makedirs(logs_name, exist_ok=True)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args['prefix']}_{args['seed']}_{args['convnet_type']}_{time_str}"
    csv_name = run_name
    logfilename = f"{logs_name}/{run_name}"
    args["time_str"] = time_str
    args["csv_name"] = csv_name
    args["logfilename"] = logfilename

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"]
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": []}
    nme_curve = {"top1": []}

    for task in range(data_manager.nb_tasks):
        logging.info("All params: %s", count_parameters(model._network))
        logging.info("Trainable params: %s", count_parameters(model._network, True))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        cnn_curve["top1"].append(cnn_accy["top1"])
        logging.info("CNN curve: %s", cnn_curve["top1"])
        if nme_accy is not None:
            nme_curve["top1"].append(nme_accy["top1"])
            logging.info("NME curve: %s", nme_curve["top1"])

        logging.info(
            "Task %s summary: CNN top1 %.2f%s",
            task,
            cnn_accy["top1"],
            f", NME top1 {nme_accy['top1']:.2f}" if nme_accy is not None else "",
        )


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if str(device) == "-1":
            gpus.append(torch.device("cpu"))
        else:
            gpus.append(torch.device(f"cuda:{device}"))

    args["device"] = gpus


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    logging.info(json.dumps({k: str(v) for k, v in args.items()}, indent=2, sort_keys=True))
