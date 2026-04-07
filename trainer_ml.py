import json
import logging
import os
import sys
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch

from utils import factory
from utils.data_manager_ml import DataManager
from utils.toolkit import count_parameters

DEFAULT_LAMBDA_LE_LIST = [
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    2,
    5,
    10,
]
DEFAULT_LAMBDA_KD_RELATION_DATA_LIST = [1]
DEFAULT_LAMBDA_KD_RELATION_AFF_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def finished(records, lamda_le, lamda_kd_relation_aff, lamda_kd_relation_data):
    if not records.get("lamda_le"):
        return False

    result = np.column_stack(
        [records["lamda_le"], records["lamda_kd_ra"], records["lamda_kd_rd"]]
    )
    query = np.array([lamda_le, lamda_kd_relation_aff, lamda_kd_relation_data])
    comparison = np.equal(result, query)
    return np.any(np.all(comparison, axis=1))


def train(args):
    seed_list = args["seed"] if isinstance(args["seed"], list) else [args["seed"]]
    device_cfg = list(args["device"]) if isinstance(args["device"], list) else [args["device"]]

    if args.get("sweep_hparams", False):
        _train_clif_sweep(args, seed_list, device_cfg)
        return

    for seed in seed_list:
        run_args = dict(args)
        run_args["seed"] = seed
        run_args["device"] = list(device_cfg)
        _train(run_args)


def _train_clif_sweep(args, seed_list, device_cfg):
    if args["model_name"].lower() != "clif":
        raise ValueError("Hyper-parameter sweep is only supported for CLIF.")

    results_dir = _get_results_dir(args)
    os.makedirs(results_dir, exist_ok=True)
    sensitivity_path = os.path.join(results_dir, "data_sensitivity.csv")

    if os.path.exists(sensitivity_path):
        records = pd.read_csv(sensitivity_path).to_dict(orient="list")
    else:
        records = {
            "lamda_le": [],
            "lamda_kd_ra": [],
            "lamda_kd_rd": [],
            "map": [],
            "macrof1": [],
            "microf1": [],
        }

    lambda_le_list = args.get("sweep_lamda_le", DEFAULT_LAMBDA_LE_LIST)
    lamda_kd_relation_data_list = args.get(
        "sweep_lamda_kd_relation_data", DEFAULT_LAMBDA_KD_RELATION_DATA_LIST
    )
    lamda_kd_relation_aff_list = args.get(
        "sweep_lamda_kd_relation_aff", DEFAULT_LAMBDA_KD_RELATION_AFF_LIST
    )

    for seed in seed_list:
        for lamda_le in lambda_le_list:
            for lamda_kd_relation_data in lamda_kd_relation_data_list:
                for lamda_kd_relation_aff in lamda_kd_relation_aff_list:
                    if finished(
                        records,
                        lamda_le,
                        lamda_kd_relation_aff,
                        lamda_kd_relation_data,
                    ):
                        continue

                    run_args = dict(args)
                    run_args["seed"] = seed
                    run_args["device"] = list(device_cfg)
                    run_args["lamda_le"] = lamda_le
                    run_args["lamda_kd_relation_data"] = lamda_kd_relation_data
                    run_args["lamda_kd_relation_aff"] = lamda_kd_relation_aff

                    map_score, macrof1, microf1 = _train(run_args)
                    records["lamda_le"].append(lamda_le)
                    records["lamda_kd_ra"].append(lamda_kd_relation_aff)
                    records["lamda_kd_rd"].append(lamda_kd_relation_data)
                    records["map"].append(map_score)
                    records["macrof1"].append(macrof1)
                    records["microf1"].append(microf1)
                    pd.DataFrame(data=records).to_csv(sensitivity_path, index=False)


def data_prepare(args):
    dataset = args["dataset"]
    data_root = args.get("data_root", "/nfs/diskstation/DataStation/KaichengFu/CIL_data")
    aux_root = args.get("aux_root", "/nfs/diskstation/DataStation/KaichengFu")

    if dataset == "iScience":
        subject = args["subject"]
        if subject == "visual":
            feature_path = args.get(
                "feature_path", os.path.join(data_root, "iScience", "v_f.npy")
            )
            sub_data = np.load(feature_path).astype("float32")
        else:
            feature_path = args.get(
                "feature_path",
                os.path.join(
                    data_root,
                    "iScience",
                    f"Subject{subject}",
                    f"data_vox8_sub{subject}.mat",
                ),
            )
            with h5py.File(feature_path, "r") as feature_file:
                sub_data = np.array(feature_file["data_vox_K"]).reshape(-1, 2196)
        sub_data = np.transpose(sub_data).astype("float32")
        args["input_size"] = sub_data.shape[1]

        label_session_path = args.get(
            "label_session_path",
            os.path.join(
                data_root,
                "iScience",
                "thre_0.1",
                _get_label_session_name(dataset, args["init_cls"], args["increment"]),
            ),
        )
        label = h5py.File(label_session_path, "r")

        dimension_path = args.get(
            "dimension_path",
            os.path.join(aux_root, "nips_data", "label_dimension.mat"),
        )
        with h5py.File(dimension_path, "r") as dimension_file:
            dimension_label = np.transpose(dimension_file["label_dimension"]).astype(
                "float32"
            )
    elif dataset == "PNAS":
        feature_path = args.get(
            "feature_path", os.path.join(data_root, "PNAS", "feature_res18.npy")
        )
        sub_data = np.load(feature_path).astype("float32")
        args["input_size"] = sub_data.shape[1]

        label_session_path = args.get(
            "label_session_path",
            os.path.join(
                data_root,
                "PNAS",
                _get_label_session_name(dataset, args["init_cls"], args["increment"]),
            ),
        )
        label = h5py.File(label_session_path, "r")

        dimension_path = args.get(
            "dimension_path",
            os.path.join(data_root, "PNAS", "affective_rating_USA.mat"),
        )
        with h5py.File(dimension_path, "r") as dimension_file:
            dimension_label = np.transpose(dimension_file["affective_rating"]).astype(
                "float32"
            )
    elif dataset == "EMOTIC":
        feature_path = args.get(
            "feature_path", os.path.join(data_root, "feature_resnet18_bbox.npy")
        )
        sub_data = np.load(feature_path).astype("float32")
        args["input_size"] = sub_data.shape[1]

        label_session_path = args.get(
            "label_session_path",
            os.path.join(
                data_root,
                _get_label_session_name(dataset, args["init_cls"], args["increment"]),
            ),
        )
        label = h5py.File(label_session_path, "r")

        dimension_path = args.get(
            "dimension_path", os.path.join(data_root, "affective_dimension.npy")
        )
        dimension_label = np.load(dimension_path).astype("float32")
    else:
        raise ValueError(f"Unsupported multi-label dataset: {dataset}")

    return sub_data, label, dimension_label


def _get_label_session_name(dataset, init_cls, increment):
    mapping = {
        "iScience": {
            (15, 3): "label_session_b15t4c3.mat",
            (15, 2): "label_session_b15t6c2.mat",
            (3, 3): "label_session_t9c3.mat",
            (9, 9): "label_session_t3c9.mat",
        },
        "PNAS": {
            (7, 4): "label_session_t7c4.mat",
            (4, 7): "label_session_t4c7.mat",
            (16, 3): "label_session_b16t4c3.mat",
            (16, 2): "label_session_b16t6c2.mat",
        },
    }

    try:
        return mapping[dataset][(init_cls, increment)]
    except KeyError as exc:
        if dataset == "EMOTIC":
            return f"label_session_b{init_cls}i{increment}.mat"
        raise ValueError(
            f"Unsupported label-session split for {dataset}: init_cls={init_cls}, increment={increment}"
        ) from exc


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
    data_all = data_prepare(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["init_cls"],
        args["increment"],
        data_all,
        total_class=args.get("total_class"),
    )
    model = factory.get_model(args["model_name"], args)

    map_curve = {"map": []}
    hamming_loss_curve = {"hamming_loss": []}
    avg_precision_curve = {"avg_precision": []}
    one_error_curve = {"one_error": []}
    ranking_loss_curve = {"ranking_loss": []}
    coverage_curve = {"coverage": []}
    macrof1_curve = {"macrof1": []}
    microf1_curve = {"microf1": []}

    for task in range(data_manager.nb_tasks):
        logging.info("All params: %s", count_parameters(model._network))
        logging.info("Trainable params: %s", count_parameters(model._network, True))
        model.incremental_train(data_manager)

        if args["model_name"] == "clif":
            test_map, test_other_metrics = model.eval_multi_label_task(clif=True)
        elif args["model_name"] == "agcn":
            test_map, test_other_metrics = model.eval_multi_label_task(agcn=True)
        else:
            test_map, test_other_metrics = model.eval_multi_label_task()

        test_map = float(test_map.cpu().detach())
        model.after_task()

        map_curve["map"].append(test_map)
        hamming_loss_curve["hamming_loss"].append(test_other_metrics[0][1])
        avg_precision_curve["avg_precision"].append(test_other_metrics[1][1])
        one_error_curve["one_error"].append(test_other_metrics[2][1])
        ranking_loss_curve["ranking_loss"].append(test_other_metrics[3][1])
        coverage_curve["coverage"].append(test_other_metrics[4][1])
        macrof1_curve["macrof1"].append(test_other_metrics[5][1])
        microf1_curve["microf1"].append(test_other_metrics[6][1])

        logging.info("Map curve: %s", map_curve["map"])
        logging.info("Hamming_loss curve: %s", hamming_loss_curve["hamming_loss"])
        logging.info("avg_pre curve: %s", avg_precision_curve["avg_precision"])
        logging.info("One_error curve: %s", one_error_curve["one_error"])
        logging.info("Ranking_loss curve: %s", ranking_loss_curve["ranking_loss"])
        logging.info("Coverage curve: %s", coverage_curve["coverage"])
        logging.info("Macrof1 curve: %s", macrof1_curve["macrof1"])
        logging.info("Microf1 curve: %s", microf1_curve["microf1"])

        logging.info(
            "Average Accuracy: %.6f, Average HL: %.6f, Average eAP: %.6f, Average OneE: %.6f, Average RL: %.6f, Average coverage: %.6f, Average macrof1: %.6f, Average microf1: %.6f",
            sum(map_curve["map"]) / len(map_curve["map"]),
            sum(hamming_loss_curve["hamming_loss"]) / len(hamming_loss_curve["hamming_loss"]),
            sum(avg_precision_curve["avg_precision"]) / len(avg_precision_curve["avg_precision"]),
            sum(one_error_curve["one_error"]) / len(one_error_curve["one_error"]),
            sum(ranking_loss_curve["ranking_loss"]) / len(ranking_loss_curve["ranking_loss"]),
            sum(coverage_curve["coverage"]) / len(coverage_curve["coverage"]),
            sum(macrof1_curve["macrof1"]) / len(macrof1_curve["macrof1"]),
            sum(microf1_curve["microf1"]) / len(microf1_curve["microf1"]),
        )

    all_result = np.zeros((8, data_manager.nb_tasks + 1))
    all_result[0, 0:data_manager.nb_tasks] = np.array(map_curve["map"])
    all_result[1, 0:data_manager.nb_tasks] = np.array(hamming_loss_curve["hamming_loss"])
    all_result[2, 0:data_manager.nb_tasks] = np.array(avg_precision_curve["avg_precision"])
    all_result[3, 0:data_manager.nb_tasks] = np.array(one_error_curve["one_error"])
    all_result[4, 0:data_manager.nb_tasks] = np.array(ranking_loss_curve["ranking_loss"])
    all_result[5, 0:data_manager.nb_tasks] = np.array(coverage_curve["coverage"])
    all_result[6, 0:data_manager.nb_tasks] = np.array(macrof1_curve["macrof1"])
    all_result[7, 0:data_manager.nb_tasks] = np.array(microf1_curve["microf1"])
    all_result[:, data_manager.nb_tasks] = np.mean(
        all_result[:, : data_manager.nb_tasks], axis=1
    )

    if args.get("save_results", True):
        _save_results(args, all_result)

    if args["model_name"] == "clif":
        return (
            float(all_result[0, data_manager.nb_tasks]),
            float(all_result[6, data_manager.nb_tasks]),
            float(all_result[7, data_manager.nb_tasks]),
        )

    return None


def _save_results(args, all_result):
    results_dir = _get_results_dir(args)
    os.makedirs(results_dir, exist_ok=True)

    if args["model_name"] == "clif":
        filename = (
            f"clif_lamda_le_{args['lamda_le']}_"
            f"lamda_kd_relation_aff_{args['lamda_kd_relation_aff']}_"
            f"lamda_kd_relation_data_{args['lamda_kd_relation_data']}.csv"
        )
    elif args["model_name"] == "replay":
        filename = f"replay_{args['buffer_type']}.csv"
    else:
        filename = f"{args['model_name']}.csv"

    np.savetxt(os.path.join(results_dir, filename), all_result, delimiter=",")


def _get_results_dir(args):
    results_root = args.get("results_root", "./results")
    init_cls = args["init_cls"]
    increment = args["increment"]

    if init_cls == 9:
        split_name = "B9I9"
    elif init_cls == 3:
        split_name = "B3I3"
    elif init_cls == 15:
        split_name = f"B15I{increment}"
    elif init_cls == 4:
        split_name = "B4I4"
    elif init_cls == 7:
        split_name = "B7I7"
    elif init_cls == 16 and increment in {2, 3}:
        split_name = f"B16I{increment}"
    else:
        split_name = f"B{init_cls}I{increment}"

    return os.path.join(results_root, args["subject"], split_name)


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
