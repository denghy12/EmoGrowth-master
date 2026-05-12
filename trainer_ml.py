import json
import logging
import os
import sys
import html
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
EMOTIC_FEATURE_SPECS = {
    "feature_vit_b_16_bbox.npy": ("vit_b_16", 768),
    "feature_resnet18_bbox.npy": ("resnet18", 512),
}
PER_CLASS_TABLE_METRICS = [
    "ap",
    "f1",
    "precision",
    "recall",
    "support",
    "predicted_positive",
    "tp",
    "fp",
    "fn",
    "tn",
]


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
    # 一个配置里可以写多个随机种子或多个设备；这里会把它们展开成多次独立实验。
    # 当前 EMOTIC 配置里 seed=[1993]，所以只会跑一次。
    seed_list = args["seed"] if isinstance(args["seed"], list) else [args["seed"]]
    device_cfg = list(args["device"]) if isinstance(args["device"], list) else [args["device"]]

    # sweep_hparams=false 表示只跑当前这组超参数；如果设为 true，
    # 会网格搜索 CLIF 里的三个关键 loss 权重。
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


def _infer_emotic_backbone(feature_path):
    basename = os.path.basename(feature_path)
    spec = EMOTIC_FEATURE_SPECS.get(basename)
    if spec is not None:
        return spec

    lowered = basename.lower()
    if "vit" in lowered:
        return "vit_b_16", 768
    if "resnet18" in lowered:
        return "resnet18", 512
    return None, None


def _validate_emotic_backbone_config(args, feature_path, feature_dim):
    inferred_backbone, expected_input_size = _infer_emotic_backbone(feature_path)
    basename = os.path.basename(feature_path)
    subject = str(args.get("subject", ""))
    subject_lower = subject.lower()
    configured_input_size = args.get("input_size")
    errors = []

    if inferred_backbone == "vit_b_16" and "vit" not in subject_lower:
        errors.append(
            "subject '{}' must include 'vit' when feature_path points to '{}'.".format(
                subject, basename
            )
        )
    elif inferred_backbone == "resnet18" and "resnet18" not in subject_lower:
        errors.append(
            "subject '{}' must include 'resnet18' when feature_path points to '{}'.".format(
                subject, basename
            )
        )

    if "vit" in subject_lower and inferred_backbone not in {None, "vit_b_16"}:
        errors.append(
            "subject '{}' implies ViT features, but feature_path points to '{}'.".format(
                subject, basename
            )
        )
    if "resnet18" in subject_lower and inferred_backbone not in {None, "resnet18"}:
        errors.append(
            "subject '{}' implies ResNet18 features, but feature_path points to '{}'.".format(
                subject, basename
            )
        )

    if expected_input_size is not None and configured_input_size is not None:
        if configured_input_size != expected_input_size:
            errors.append(
                "configured input_size={} does not match expected input_size={} for '{}'.".format(
                    configured_input_size, expected_input_size, basename
                )
            )

    if expected_input_size is not None and feature_dim != expected_input_size:
        errors.append(
            "loaded feature dimension {} does not match expected input_size {} for '{}'.".format(
                feature_dim, expected_input_size, basename
            )
        )

    if errors:
        raise ValueError(
            "Invalid EMOTIC backbone configuration:\n- {}".format("\n- ".join(errors))
        )


def data_prepare(args):
    # 把配置里的数据相关字段读成三个对象：
    # sub_data：样本特征；label：每个增量任务的训练/测试标签；
    # dimension_label：情感维度向量，后面 CLIF 会拿它做关系蒸馏。
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
        # EMOTIC 的 ViT 配置使用提前抽好的 ViT-B/16 bbox 特征。
        # 也就是说模型训练的输入是 768 维特征向量，不是原始图片。
        feature_path = args.get(
            "feature_path", os.path.join(data_root, "feature_vit_b_16_bbox.npy")
        )
        sub_data = np.load(feature_path).astype("float32")
        _validate_emotic_backbone_config(args, feature_path, sub_data.shape[1])
        args["input_size"] = sub_data.shape[1]

        # init_cls=5、increment=3 会选择 label_session_b5i3.mat。
        # 这就是 B5I3 协议：先学 5 个情绪类别，之后每轮新增 3 个类别。
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
    # prefix、seed、subject、convnet_type 会组成本次运行的名字，
    # 这样日志和 CSV 文件能追溯到具体实验设置。
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    subject_name = str(args.get("subject", "default")).replace("/", "_")
    split_name = _get_split_name(args["init_cls"], args["increment"])
    logs_name = _get_logs_dir(args, split_name)
    os.makedirs(logs_name, exist_ok=True)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args['prefix']}_{args['seed']}_{subject_name}_{args['convnet_type']}_{time_str}"
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

    # DataManager 负责管理“增量任务划分”。
    # 对 EMOTIC 来说 total_class=26，B5I3 会形成 [5, 3, 3, ..., 最后一批]。
    data_manager = DataManager(
        args["dataset"],
        args["init_cls"],
        args["increment"],
        data_all,
        total_class=args.get("total_class"),
    )
    # model_name="clif" 会通过 utils/factory.py 实例化 models/clif_ml.py 里的 CLIF。
    model = factory.get_model(args["model_name"], args)
    save_detailed_results = _detailed_results_enabled(args)
    detail_context = None
    if save_detailed_results:
        detail_context = _build_detail_context(args, data_manager)
        _prepare_detail_result_files(args)
        detail_paths = _get_detail_result_paths(args)
        logging.info("Detailed per-class metrics: %s", detail_paths["per_class"])
        logging.info("Detailed per-task metrics: %s", detail_paths["per_task"])
        logging.info("Detailed per-class task table: %s", detail_paths["per_class_table"])
        logging.info(
            "Detailed per-class task HTML: %s", detail_paths["per_class_table_html"]
        )
        logging.info("Detailed loss metrics: %s", detail_paths["loss"])
        args["loss_detail_path"] = detail_paths["loss"]

    map_curve = {"map": []}
    hamming_loss_curve = {"hamming_loss": []}
    avg_precision_curve = {"avg_precision": []}
    one_error_curve = {"one_error": []}
    ranking_loss_curve = {"ranking_loss": []}
    coverage_curve = {"coverage": []}
    macrof1_curve = {"macrof1": []}
    microf1_curve = {"microf1": []}

    for task in range(data_manager.nb_tasks):
        # 每次循环就是一个增量任务：先训练当前新增类别，再在当前已见类别上评估，
        # 最后把模型冻结成 old model，供下一轮做知识蒸馏。
        logging.info("All params: %s", count_parameters(model._network))
        logging.info("Trainable params: %s", count_parameters(model._network, True))
        model.incremental_train(data_manager)

        if args["model_name"] == "clif":
            eval_result = model.eval_multi_label_task(
                clif=True, return_outputs=save_detailed_results
            )
        elif args["model_name"] == "agcn":
            eval_result = model.eval_multi_label_task(
                agcn=True, return_outputs=save_detailed_results
            )
        else:
            eval_result = model.eval_multi_label_task(
                return_outputs=save_detailed_results
            )

        if save_detailed_results:
            test_map, test_other_metrics, detail_outputs, detail_labels = eval_result
            _append_detail_results(
                args,
                detail_context,
                task,
                detail_outputs,
                detail_labels,
            )
        else:
            test_map, test_other_metrics = eval_result

        test_map = _to_float(test_map)
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
    np.savetxt(
        os.path.join(results_dir, _get_result_filename(args)),
        all_result,
        delimiter=",",
    )


def _get_result_filename(args):
    variant = args.get("method_variant")
    variant = str(variant).strip() if variant else ""

    if args["model_name"] == "clif":
        clif_variant = f"{variant}_" if variant else ""
        return (
            f"clif_{clif_variant}lamda_le_{args['lamda_le']}_"
            f"lamda_kd_relation_aff_{args['lamda_kd_relation_aff']}_"
            f"lamda_kd_relation_data_{args['lamda_kd_relation_data']}.csv"
        )
    if args["model_name"] == "replay":
        replay_name = f"replay_{args['buffer_type']}"
        return f"{replay_name}_{variant}.csv" if variant else f"{replay_name}.csv"
    if variant:
        return f"{args['model_name']}_{variant}.csv"
    return f"{args['model_name']}.csv"


def _truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _detailed_results_enabled(args):
    return _truthy(args.get("save_detailed_results", False))


def _to_float(value):
    if hasattr(value, "cpu"):
        return float(value.cpu().detach())
    return float(value)


def _get_detail_results_dir(args):
    split_name = _get_split_name(args["init_cls"], args["increment"])

    if args.get("dataset") == "EMOTIC":
        feature_path = args.get("feature_path", "")
        inferred_backbone, _ = _infer_emotic_backbone(feature_path)
        extractor_name = "vit" if inferred_backbone == "vit_b_16" else "resnet18"
        order_name, protocol_name, _ = _infer_emotic_order_and_protocol(args)
        loss_name = _infer_emotic_loss_name(args)
        scale_name = _infer_emotic_scale(args)
        result_root = args.get("detail_result_root", "./result_detail")
        return os.path.join(
            result_root,
            extractor_name,
            order_name,
            protocol_name,
            split_name,
            loss_name,
            scale_name,
        )

    detail_root = args.get("detail_result_root", "./result_detail")
    return os.path.join(detail_root, args["subject"], split_name)


def _get_detail_result_paths(args):
    detail_dir = _get_detail_results_dir(args)
    run_name = args.get("csv_name", args.get("prefix", "run"))
    return {
        "per_class": os.path.join(detail_dir, f"{run_name}_per_class_metrics.csv"),
        "per_task": os.path.join(detail_dir, f"{run_name}_per_task_metrics.csv"),
        "per_class_table": os.path.join(
            detail_dir, f"{run_name}_per_class_task_table.csv"
        ),
        "per_class_table_html": os.path.join(
            detail_dir, f"{run_name}_per_class_task_table.html"
        ),
        "loss": os.path.join(detail_dir, f"{run_name}_loss_metrics.csv"),
    }


def _prepare_detail_result_files(args):
    paths = _get_detail_result_paths(args)
    os.makedirs(os.path.dirname(paths["per_class"]), exist_ok=True)
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)


def _build_detail_context(args, data_manager):
    _, _, manifest = _infer_emotic_order_and_protocol(args)
    task_groups = manifest.get("task_groups", []) if isinstance(manifest, dict) else []
    class_names = []
    for task_group in task_groups:
        class_names.extend(task_group.get("classes", []))
    if not class_names:
        class_order_path = os.path.join(args.get("data_root", ""), "class_order.json")
        class_order = _load_json_if_exists(class_order_path)
        if isinstance(class_order, list):
            class_names.extend(class_order)

    total_classes = data_manager.get_total_classnum()
    while len(class_names) < total_classes:
        class_names.append(f"class_{len(class_names)}")

    task_ranges = []
    class_task_ids = np.full(total_classes, -1, dtype=int)
    start = 0
    for task_id, task_size in enumerate(data_manager._increments):
        end = start + task_size
        task_ranges.append((start, end))
        class_task_ids[start:end] = task_id
        start = end

    return {
        "class_names": class_names,
        "class_task_ids": class_task_ids,
        "task_ranges": task_ranges,
    }


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _class_average_precision(scores, targets):
    targets = targets.astype(int)
    support = int(np.sum(targets == 1))
    if support == 0:
        return np.nan

    order = np.argsort(-scores)
    sorted_targets = targets[order]
    positives_seen = np.cumsum(sorted_targets == 1)
    ranks = np.arange(1, len(sorted_targets) + 1)
    precision_at_positive = positives_seen[sorted_targets == 1] / ranks[sorted_targets == 1]
    return float(np.mean(precision_at_positive))


def _sample_average_precision(outputs, labels):
    valid_scores = []
    for scores, targets in zip(outputs, labels):
        positive_count = int(np.count_nonzero(targets == 1))
        if positive_count == 0:
            continue
        order = np.argsort(-scores)
        sorted_targets = targets[order]
        positives_seen = np.cumsum(sorted_targets == 1)
        ranks = np.arange(1, len(sorted_targets) + 1)
        precision_at_positive = positives_seen[sorted_targets == 1] / ranks[sorted_targets == 1]
        valid_scores.append(float(np.mean(precision_at_positive)))
    return float(np.mean(valid_scores)) if valid_scores else np.nan


def _ranking_loss(outputs, labels):
    losses = []
    for scores, targets in zip(outputs, labels):
        positive_count = int(np.count_nonzero(targets == 1))
        negative_count = targets.shape[0] - positive_count
        if positive_count == 0 or negative_count == 0:
            continue
        order = np.argsort(-scores)
        sorted_targets = targets[order]
        negatives_seen = 0
        inversions = 0
        for target in sorted_targets:
            if target == 0:
                negatives_seen += 1
            else:
                inversions += negatives_seen
        losses.append(inversions / (positive_count * negative_count))
    return float(np.mean(losses)) if losses else np.nan


def _coverage(outputs, labels):
    coverage_values = []
    class_count = labels.shape[1]
    if class_count == 0:
        return np.nan
    for scores, targets in zip(outputs, labels):
        if np.sum(targets == 1) == 0:
            continue
        order = np.argsort(-scores)
        sorted_targets = targets[order]
        last_positive_rank = int(np.max(np.where(sorted_targets == 1)))
        coverage_values.append(last_positive_rank / class_count)
    return float(np.mean(coverage_values)) if coverage_values else np.nan


def _one_error(outputs, labels):
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        return np.nan
    top_indices = np.argmax(outputs, axis=1)
    misses = labels[np.arange(labels.shape[0]), top_indices] != 1
    return float(np.mean(misses))


def _per_class_rows(outputs, labels, detail_context, trained_task):
    outputs = np.asarray(outputs)
    labels = np.asarray(labels).astype(int)
    predictions = (outputs > 0).astype(int)
    width = min(outputs.shape[1], labels.shape[1], len(detail_context["class_names"]))
    rows = []

    for class_id in range(width):
        target = labels[:, class_id]
        prediction = predictions[:, class_id]
        tp = int(np.sum((prediction == 1) & (target == 1)))
        fp = int(np.sum((prediction == 1) & (target == 0)))
        fn = int(np.sum((prediction == 0) & (target == 1)))
        tn = int(np.sum((prediction == 0) & (target == 0)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "trained_task": trained_task,
                "trained_seen_classes": width,
                "class_id": class_id,
                "class_name": detail_context["class_names"][class_id],
                "class_task": int(detail_context["class_task_ids"][class_id]),
                "ap": _class_average_precision(outputs[:, class_id], target),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "support": int(np.sum(target == 1)),
                "predicted_positive": int(np.sum(prediction == 1)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return rows


def _slice_metrics(outputs, labels):
    outputs = np.asarray(outputs)
    labels = np.asarray(labels).astype(int)
    if outputs.shape[0] == 0 or outputs.shape[1] == 0:
        return {
            "map": np.nan,
            "hamming_loss": np.nan,
            "avg_precision": np.nan,
            "one_error": np.nan,
            "ranking_loss": np.nan,
            "coverage": np.nan,
            "macrof1": np.nan,
            "microf1": np.nan,
        }

    predictions = (outputs > 0).astype(int)
    tp = np.sum((predictions == 1) & (labels == 1), axis=0)
    fp = np.sum((predictions == 1) & (labels == 0), axis=0)
    fn = np.sum((predictions == 0) & (labels == 1), axis=0)

    class_f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) != 0,
    )
    total_tp = int(np.sum(tp))
    total_fp = int(np.sum(fp))
    total_fn = int(np.sum(fn))
    per_class_ap = [
        _class_average_precision(outputs[:, class_id], labels[:, class_id])
        for class_id in range(labels.shape[1])
    ]
    valid_ap = [ap for ap in per_class_ap if not np.isnan(ap)]

    return {
        "map": float(np.mean(valid_ap)) if valid_ap else np.nan,
        "hamming_loss": float(np.mean(predictions != labels)),
        "avg_precision": _sample_average_precision(outputs, labels),
        "one_error": _one_error(outputs, labels),
        "ranking_loss": _ranking_loss(outputs, labels),
        "coverage": _coverage(outputs, labels),
        "macrof1": float(np.mean(class_f1)),
        "microf1": _safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn),
    }


def _per_task_rows(outputs, labels, detail_context, trained_task):
    outputs = np.asarray(outputs)
    labels = np.asarray(labels).astype(int)
    width = min(outputs.shape[1], labels.shape[1])
    rows = []

    for eval_task, (start, end) in enumerate(detail_context["task_ranges"]):
        if start >= width or eval_task > trained_task:
            continue
        clipped_end = min(end, width)
        task_labels = labels[:, start:clipped_end]
        task_outputs = outputs[:, start:clipped_end]
        positive_mask = np.sum(task_labels == 1, axis=1) > 0
        filtered_outputs = task_outputs[positive_mask]
        filtered_labels = task_labels[positive_mask]
        metrics = _slice_metrics(filtered_outputs, filtered_labels)
        row = {
            "trained_task": trained_task,
            "trained_seen_classes": width,
            "eval_task": eval_task,
            "class_start": start,
            "class_end_exclusive": clipped_end,
            "num_classes": clipped_end - start,
            "num_samples": int(np.sum(positive_mask)),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _per_class_task_table(per_class_df, metrics=None):
    if per_class_df.empty:
        return pd.DataFrame(columns=["class_id", "class_name", "class_task"])

    metrics = metrics or PER_CLASS_TABLE_METRICS
    metrics = [metric for metric in metrics if metric in per_class_df.columns]
    if "trained_task" not in per_class_df.columns or not metrics:
        return per_class_df.copy()

    table_df = per_class_df.copy()
    table_df["trained_task"] = pd.to_numeric(
        table_df["trained_task"], errors="coerce"
    )
    table_df = table_df.dropna(subset=["trained_task"])
    table_df["trained_task"] = table_df["trained_task"].astype(int)

    key_col = "class_id" if "class_id" in table_df.columns else "class_name"
    index_cols = [
        column
        for column in ["class_id", "class_name", "class_task"]
        if column in table_df.columns
    ]
    base = table_df[index_cols].drop_duplicates(subset=[key_col]).copy()
    if "class_id" in base.columns:
        base = base.sort_values("class_id")
    elif "class_name" in base.columns:
        base = base.sort_values("class_name")
    base = base.reset_index(drop=True)

    tasks = sorted(table_df["trained_task"].unique())
    for task in tasks:
        task_df = table_df[table_df["trained_task"] == task].drop_duplicates(
            subset=[key_col], keep="last"
        )
        task_df = task_df.set_index(key_col)
        for metric in metrics:
            base[f"task{task}_{metric}"] = base[key_col].map(task_df[metric])

    return base


def _task_metric_column_sort_key(column):
    task_prefix = str(column).split("_", 1)[0]
    if task_prefix.startswith("task") and task_prefix[4:].isdigit():
        return int(task_prefix[4:])
    return 10**9


def _add_count_column(table_df):
    display_df = table_df.copy()
    if "count" in display_df.columns:
        return display_df
    if "次数" in display_df.columns:
        return display_df.rename(columns={"次数": "count"})

    support_columns = sorted(
        [
            column
            for column in display_df.columns
            if str(column).startswith("task") and str(column).endswith("_support")
        ],
        key=_task_metric_column_sort_key,
    )
    count_values = (
        display_df[support_columns].bfill(axis=1).iloc[:, 0]
        if support_columns
        else None
    )
    display_df.insert(min(3, len(display_df.columns)), "count", count_values)
    return display_df


def _write_per_class_task_html(table_df, html_path, title):
    html_title = html.escape(title)
    display_df = _add_count_column(table_df)
    table_for_json = display_df.astype(object).where(pd.notna(display_df), None)
    columns_json = json.dumps(list(table_for_json.columns), ensure_ascii=False)
    rows_json = json.dumps(
        table_for_json.to_dict(orient="records"), ensure_ascii=False
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html_title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: 28px;
      line-height: 1.2;
    }}
    .toolbar {{
      align-items: center;
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .segmented-control {{
      display: inline-flex;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      overflow: hidden;
      background: #ffffff;
    }}
    .segmented-control button {{
      border: 0;
      border-right: 1px solid #cbd5e1;
      background: #ffffff;
      color: #334155;
      cursor: pointer;
      font-size: 13px;
      padding: 7px 12px;
    }}
    .segmented-control button:last-child {{
      border-right: 0;
    }}
    .segmented-control button.active {{
      background: #0f172a;
      color: #ffffff;
    }}
    .layout-hint {{
      color: #64748b;
      font-size: 13px;
    }}
    .table-wrap {{
      max-height: calc(100vh - 130px);
      overflow: auto;
      border: 1px solid #d1d5db;
    }}
    table.metric-table {{
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }}
    .metric-table th,
    .metric-table td {{
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: right;
    }}
    .metric-table th {{
      position: sticky;
      top: 0;
      background: #f3f4f6;
      z-index: 2;
    }}
    .metric-table th:nth-child(1),
    .metric-table td:nth-child(1) {{
      position: sticky;
      left: 0;
      min-width: 64px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(2),
    .metric-table td:nth-child(2) {{
      position: sticky;
      left: 64px;
      min-width: 150px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(3),
    .metric-table td:nth-child(3) {{
      position: sticky;
      left: 214px;
      min-width: 80px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(4),
    .metric-table td:nth-child(4) {{
      position: sticky;
      left: 294px;
      min-width: 80px;
      text-align: right;
      background: #ffffff;
      z-index: 1;
      box-shadow: 1px 0 0 #e5e7eb;
    }}
    .metric-table th:nth-child(-n+4) {{
      background: #f3f4f6;
      z-index: 3;
    }}
  </style>
</head>
<body>
  <h1>{html_title}</h1>
  <div class="toolbar">
    <div class="segmented-control" role="group" aria-label="column layout">
      <button type="button" class="active" data-layout="task">按 task 分组</button>
      <button type="button" data-layout="metric">按指标分组</button>
    </div>
    <span class="layout-hint" id="layoutHint"></span>
  </div>
  <div class="table-wrap">
    <table class="metric-table" id="metricTable"></table>
  </div>
  <script>
    const tableColumns = {columns_json};
    const tableRows = {rows_json};
    const metricPreference = {json.dumps(PER_CLASS_TABLE_METRICS, ensure_ascii=False)};
    const frozenColumnCount = 4;
    const indexColumns = tableColumns.slice(0, frozenColumnCount);
    const metricColumns = tableColumns.slice(frozenColumnCount);

    function parseMetricColumn(column) {{
      const match = /^task(\\d+)_(.+)$/.exec(column);
      if (!match) {{
        return null;
      }}
      return {{
        column,
        task: Number(match[1]),
        metric: match[2],
      }};
    }}

    const parsedColumns = metricColumns
      .map(parseMetricColumn)
      .filter(Boolean);
    const tasks = [...new Set(parsedColumns.map((item) => item.task))]
      .sort((a, b) => a - b);
    const discoveredMetrics = [...new Set(parsedColumns.map((item) => item.metric))];
    const metrics = [
      ...metricPreference.filter((metric) => discoveredMetrics.includes(metric)),
      ...discoveredMetrics.filter((metric) => !metricPreference.includes(metric)),
    ];
    const knownMetricColumns = new Set(
      parsedColumns.map((item) => item.column)
    );
    const extraColumns = metricColumns.filter(
      (column) => !knownMetricColumns.has(column)
    );

    function taskFirstColumns() {{
      return [
        ...indexColumns,
        ...tasks.flatMap((task) =>
          metrics
            .map((metric) => `task${{task}}_${{metric}}`)
            .filter((column) => tableColumns.includes(column))
        ),
        ...extraColumns,
      ];
    }}

    function metricFirstColumns() {{
      return [
        ...indexColumns,
        ...metrics.flatMap((metric) =>
          tasks
            .map((task) => `task${{task}}_${{metric}}`)
            .filter((column) => tableColumns.includes(column))
        ),
        ...extraColumns,
      ];
    }}

    function formatValue(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return "NaN";
      }}
      if (typeof value === "number") {{
        return Number.isInteger(value) ? String(value) : value.toFixed(6);
      }}
      return String(value);
    }}

    function renderTable(columns) {{
      const table = document.getElementById("metricTable");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      columns.forEach((column) => {{
        const th = document.createElement("th");
        th.textContent = column;
        headerRow.appendChild(th);
      }});
      thead.appendChild(headerRow);

      const tbody = document.createElement("tbody");
      tableRows.forEach((row) => {{
        const tr = document.createElement("tr");
        columns.forEach((column) => {{
          const td = document.createElement("td");
          td.textContent = formatValue(row[column]);
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});

      table.replaceChildren(thead, tbody);
    }}

    function setLayout(layout) {{
      const isMetricLayout = layout === "metric";
      renderTable(isMetricLayout ? metricFirstColumns() : taskFirstColumns());
      document.querySelectorAll("[data-layout]").forEach((button) => {{
        button.classList.toggle("active", button.dataset.layout === layout);
      }});
      document.getElementById("layoutHint").textContent = isMetricLayout
        ? "当前列顺序：task0-7 的 ap，然后 task0-7 的 f1，依次类推。"
        : "当前列顺序：task0 的所有指标，然后 task1 的所有指标，依次类推。";
    }}

    document.querySelectorAll("[data-layout]").forEach((button) => {{
      button.addEventListener("click", () => setLayout(button.dataset.layout));
    }});
    setLayout("task");
  </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(content)


def _write_per_class_task_table(per_class_path, table_path, html_path=None):
    if not os.path.exists(per_class_path):
        return

    per_class_df = pd.read_csv(per_class_path)
    table_df = _per_class_task_table(per_class_df)
    table_df.to_csv(table_path, index=False)
    if html_path:
        title = os.path.basename(html_path).replace(".html", "")
        _write_per_class_task_html(table_df, html_path, title)


def _append_detail_results(args, detail_context, trained_task, outputs, labels):
    paths = _get_detail_result_paths(args)
    os.makedirs(os.path.dirname(paths["per_class"]), exist_ok=True)

    per_class_df = pd.DataFrame(
        _per_class_rows(outputs, labels, detail_context, trained_task)
    )
    per_task_df = pd.DataFrame(
        _per_task_rows(outputs, labels, detail_context, trained_task)
    )
    per_class_df.to_csv(
        paths["per_class"],
        mode="a",
        index=False,
        header=not os.path.exists(paths["per_class"]),
    )
    per_task_df.to_csv(
        paths["per_task"],
        mode="a",
        index=False,
        header=not os.path.exists(paths["per_task"]),
    )
    _write_per_class_task_table(
        paths["per_class"],
        paths["per_class_table"],
        paths["per_class_table_html"],
    )


def _get_results_dir(args):
    split_name = _get_split_name(args["init_cls"], args["increment"])

    if args.get("dataset") == "EMOTIC":
        return _get_emotic_results_dir(args, split_name)

    results_root = args.get("results_root", "./results")
    return os.path.join(results_root, args["subject"], split_name)


def _get_split_name(init_cls, increment):
    if init_cls == 9:
        return "B9I9"
    if init_cls == 3:
        return "B3I3"
    if init_cls == 15:
        return f"B15I{increment}"
    if init_cls == 4:
        return "B4I4"
    if init_cls == 7:
        return "B7I7"
    if init_cls == 16 and increment in {2, 3}:
        return f"B16I{increment}"
    return f"B{init_cls}I{increment}"


def _load_json_if_exists(path):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return None


def _infer_emotic_order_and_protocol(args):
    protocol_override = args.get("protocol_name")
    if protocol_override:
        protocol_override = str(protocol_override)
    data_root = args.get("data_root", "")
    root_name = os.path.basename(os.path.abspath(data_root)).lower()
    manifest = _load_json_if_exists(os.path.join(data_root, "label_session_manifest.json"))
    prepare_manifest = _load_json_if_exists(os.path.join(data_root, "prepare_manifest.json"))

    if manifest is not None:
        class_order_mode = manifest.get("class_order_mode", "existing")
        train_assignment_mode = manifest.get("train_assignment_mode", "repeat_current")
        train_label_mode = manifest.get("train_label_mode", args.get("train_label_mode", "current"))
    else:
        class_order_mode = None
        train_assignment_mode = None
        train_label_mode = args.get("train_label_mode", None)

    if "balanced" in root_name:
        order_name = "balanced"
    elif class_order_mode == "alphabetical" or "alpha" in root_name:
        order_name = "alphabetical"
    else:
        order_name = "frequency"

    if protocol_override:
        protocol_name = protocol_override
    elif train_assignment_mode == "strict_owner" or "strict_owner" in root_name or "strict" in root_name:
        protocol_name = "strict"
    elif str(args.get("agcnpp_protocol", "")).lower() in {"il", "cl"}:
        protocol_name = f"agcnpp_{str(args['agcnpp_protocol']).lower()}"
    else:
        protocol_name = "original"

    return order_name, protocol_name, manifest or prepare_manifest or {}


def _infer_emotic_scale(args):
    joined = " ".join(
        str(args.get(key, "")).lower()
        for key in ["prefix", "subject", "results_root", "result_root"]
    )
    return "smoke" if "smoke" in joined else "formal"


def _infer_emotic_loss_name(args):
    return str(args.get("loss_type", "softmargin")).lower()


def _get_emotic_results_dir(args, split_name):
    feature_path = args.get("feature_path", "")
    inferred_backbone, _ = _infer_emotic_backbone(feature_path)
    extractor_name = "vit" if inferred_backbone == "vit_b_16" else "resnet18"
    order_name, protocol_name, _ = _infer_emotic_order_and_protocol(args)
    loss_name = _infer_emotic_loss_name(args)
    scale_name = _infer_emotic_scale(args)
    result_root = args.get("result_root", "./result")
    return os.path.join(
        result_root,
        extractor_name,
        order_name,
        protocol_name,
        split_name,
        loss_name,
        scale_name,
    )


def _get_logs_dir(args, split_name):
    if args.get("dataset") == "EMOTIC":
        feature_path = args.get("feature_path", "")
        inferred_backbone, _ = _infer_emotic_backbone(feature_path)
        extractor_name = "vit" if inferred_backbone == "vit_b_16" else "resnet18"
        order_name, protocol_name, _ = _infer_emotic_order_and_protocol(args)
        loss_name = _infer_emotic_loss_name(args)
        scale_name = _infer_emotic_scale(args)
        return os.path.join(
            "logs",
            extractor_name,
            order_name,
            protocol_name,
            split_name,
            loss_name,
            scale_name,
            args["model_name"],
        )

    subject_name = str(args.get("subject", "default")).replace("/", "_")
    return os.path.join("logs", args["model_name"], args["dataset"], subject_name, str(args["init_cls"] if args["init_cls"] != args["increment"] else 0), str(args["increment"]))


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
