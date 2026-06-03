import sys
import csv
import json
import logging
import copy
import random
import torch
from torch.utils.data import DataLoader
import swanlab
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import time

EPSILON = 1e-8


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    # 获取分布式信息
    is_distributed = args.get("is_distributed", False)
    local_rank = args.get("local_rank", 0)
    args.pop("task_increments", None)
    args.pop("si_blurry_eval_groups", None)

    init_cls_arg = args.get("init_cls", args.get("increment", 0))
    increment_arg = args.get("increment", init_cls_arg)
    init_cls = 0 if init_cls_arg == increment_arg else init_cls_arg
    log_root = _build_log_root(args, init_cls)
    resume_path = _resolve_resume_path(args, log_root)

    # 1. 只有主进程创建文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if resume_path:
        logs_name = os.path.dirname(os.path.dirname(resume_path))
        timestamp = os.path.basename(logs_name)
    else:
        logs_name = os.path.join(log_root, timestamp)
    if local_rank <= 0:
        if not os.path.exists(logs_name):
            os.makedirs(logs_name)
    args["logs_name"] = logs_name
    args["diagnostics_dir"] = os.path.join(logs_name, "diagnostics")

    logfilename = os.path.join(
        logs_name,
        "{}_{}_{}".format(args["prefix"], args["seed"], args["convnet_type"]),
    )

    # 2. 日志配置：只有 local_rank 0 打印到控制台和文件，其他进程保持静默
    if local_rank <= 0:
        # 重置 logging 处理器，防止重复
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=logfilename + ".log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(level=logging.ERROR, handlers=[logging.NullHandler()])

    # 3. 设置随机种子
    _set_random(args["seed"])

    # 4. 设置设备
    _set_device(args)
    if local_rank <= 0:
        print_args(args)

    # 5. 初始化数据和模型
    _apply_data_root(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        init_cls_arg,
        increment_arg,
        args["aug"] if "aug" in args else 1,
        args=args,
    )
    args["class_order"] = list(getattr(data_manager, "_class_order", []))
    args["task_increments"] = data_manager.get_task_sizes()
    si_blurry_eval_groups = data_manager.get_si_blurry_eval_groups()
    if si_blurry_eval_groups is not None:
        args["si_blurry_eval_groups"] = si_blurry_eval_groups
    if local_rank <= 0:
        _save_run_reproducibility(args, logs_name)

    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    start_task = 0

    if resume_path:
        start_task, history_state = _resume_from_checkpoint(model, data_manager, resume_path)
        cnn_curve = history_state["cnn_curve"]
        nme_curve = history_state["nme_curve"]
        cnn_matrix = history_state["cnn_matrix"]
        nme_matrix = history_state["nme_matrix"]
        if local_rank <= 0:
            logging.info("Resume from checkpoint: {}".format(resume_path))
            logging.info("Resuming from task index {}".format(start_task))

    # 6. 开始任务循环
    for task in range(start_task, data_manager.nb_tasks):
        data_manager.set_current_task(task)
        if local_rank <= 0:
            logging.info("All params: {}".format(count_parameters(model._network)))
            logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        # 增量训练
        model.incremental_train(data_manager)

        ckpt_dir = os.path.join(logs_name, "checkpoints")
        if local_rank <= 0 and not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        if hasattr(model, "has_pending_final_sigreg") and model.has_pending_final_sigreg():
            history_state = _make_history_state(cnn_curve, nme_curve, cnn_matrix, nme_matrix)
            final_artifact_dir = os.path.join(logs_name, "final_sigreg")

            before_cnn_accy = None
            before_nme_accy = None
            if local_rank <= 0:
                before_cnn_accy, before_cnn_pred, before_cnn_true = _eval_cnn_outputs(model)
                logging.info("Final SIGReg before CNN: {}".format(before_cnn_accy["grouped"]))
                if args.get("record_final_sigreg_nme", False):
                    (
                        before_nme_accy,
                        before_nme_pred,
                        before_nme_true,
                        _,
                    ) = _eval_nme_with_fresh_class_means(model, data_manager)
                    logging.info(
                        "Final SIGReg before NME (analysis class means): {}".format(
                            before_nme_accy["grouped"]
                        )
                    )

                if args.get("record_confusion_matrix", False):
                    final_conf_dir = os.path.join(final_artifact_dir, "confusion_matrices")
                    _save_confusion_artifacts(
                        final_conf_dir,
                        task,
                        "before_final_sigreg",
                        "CNN",
                        before_cnn_pred,
                        before_cnn_true,
                        model._total_classes,
                    )
                    if before_nme_accy is not None:
                        _save_confusion_artifacts(
                            final_conf_dir,
                            task,
                            "before_final_sigreg",
                            "NME_analysis",
                            before_nme_pred,
                            before_nme_true,
                            model._total_classes,
                        )

                before_path = os.path.join(
                    ckpt_dir,
                    "{}_{}_task_{}_before_final_sigreg.pth".format(
                        args["prefix"], args["seed"], task
                    ),
                )
                _save_resume_checkpoint(
                    model,
                    before_path,
                    task,
                    history_state,
                    extra_state={"final_sigreg_stage": "before"},
                )
                logging.info("!!! Before final SIGReg checkpoint saved to: {} !!!".format(before_path))

            final_state = model.run_final_sigreg_calibration(
                data_manager,
                final_artifact_dir,
                ckpt_dir=ckpt_dir,
            )

            after_cnn_accy = None
            after_nme_accy = None
            if local_rank <= 0:
                after_cnn_accy, after_cnn_pred, after_cnn_true = _eval_cnn_outputs(model)
                logging.info("Final SIGReg after CNN: {}".format(after_cnn_accy["grouped"]))
                if args.get("record_final_sigreg_nme", False):
                    (
                        after_nme_accy,
                        after_nme_pred,
                        after_nme_true,
                        _,
                    ) = _eval_nme_with_fresh_class_means(model, data_manager)
                    logging.info(
                        "Final SIGReg after NME (analysis class means): {}".format(
                            after_nme_accy["grouped"]
                        )
                    )

                if args.get("record_confusion_matrix", False):
                    final_conf_dir = os.path.join(final_artifact_dir, "confusion_matrices")
                    _save_confusion_artifacts(
                        final_conf_dir,
                        task,
                        "after_final_sigreg",
                        "CNN",
                        after_cnn_pred,
                        after_cnn_true,
                        model._total_classes,
                    )
                    if after_nme_accy is not None:
                        _save_confusion_artifacts(
                            final_conf_dir,
                            task,
                            "after_final_sigreg",
                            "NME_analysis",
                            after_nme_pred,
                            after_nme_true,
                            model._total_classes,
                        )

                after_path = os.path.join(
                    ckpt_dir,
                    "{}_{}_task_{}_after_final_sigreg.pth".format(
                        args["prefix"], args["seed"], task
                    ),
                )
                _save_resume_checkpoint(
                    model,
                    after_path,
                    task,
                    history_state,
                    extra_state={
                        "final_sigreg_stage": "after",
                        "final_sigreg_state": final_state,
                    },
                )
                logging.info("!!! After final SIGReg checkpoint saved to: {} !!!".format(after_path))
                _save_final_sigreg_accuracy(
                    final_artifact_dir,
                    task,
                    before_cnn_accy,
                    after_cnn_accy,
                    before_nme_accy,
                    after_nme_accy,
                    final_state,
                )

            model.complete_incremental_train(data_manager)

        #accy是字典
        cnn_accy, nme_accy = model.eval_task()
        if local_rank <= 0 and args.get("record_confusion_matrix", False):
            _save_model_confusions(
                model,
                os.path.join(args["diagnostics_dir"], "confusion_matrices"),
                task,
                "official_after_task",
            )
        model.after_task()

        # 只有主进程收集并打印当前任务的结果
        if local_rank <= 0:
            if nme_accy is not None:
                logging.info("CNN: {}".format(cnn_accy["grouped"]))
                logging.info("NME: {}".format(nme_accy["grouped"]))

                cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                cnn_keys_sorted = sorted(cnn_keys)
                cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
                cnn_matrix.append(cnn_values)

                nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
                nme_keys_sorted = sorted(nme_keys)
                nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
                nme_matrix.append(nme_values)

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])
                nme_curve["top1"].append(nme_accy["top1"])
                nme_curve["top5"].append(nme_accy["top5"])

                logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
                logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            else:
                logging.info("No NME accuracy.")
                logging.info("CNN: {}".format(cnn_accy["grouped"]))

                cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                cnn_keys_sorted = sorted(cnn_keys)
                cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
                cnn_matrix.append(cnn_values)

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])
                logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))

            summary_payload = _task_summary_payload(
                "CNN",
                cnn_accy,
                include_old=model._known_classes > 0,
            )
            summary_payload.update(
                _task_summary_payload(
                    "NME",
                    nme_accy,
                    include_old=model._known_classes > 0,
                )
            )
            if summary_payload:
                swanlab.log(summary_payload, step=task)

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            save_path = os.path.join(ckpt_dir, "{}_{}_task_{}.pth".format(
                args["prefix"], args["seed"], task
            ))

            history_state = _make_history_state(cnn_curve, nme_curve, cnn_matrix, nme_matrix)
            _save_resume_checkpoint(model, save_path, task, history_state)
            logging.info("!!! Model checkpoint saved to: {} !!!".format(save_path))

    # 7. 训练结束，主进程汇总 Accuracy Matrix 和 Forgetting
    if local_rank <= 0:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([data_manager.nb_tasks, data_manager.nb_tasks])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:data_manager.nb_tasks - 1])
            logging.info('Accuracy Matrix (CNN):\n{}'.format(np_acctable))
            logging.info('Forgetting (CNN): {}'.format(forgetting))

        if len(nme_matrix) > 0:
            np_acctable = np.zeros([data_manager.nb_tasks, data_manager.nb_tasks])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:data_manager.nb_tasks - 1])
            logging.info('Accuracy Matrix (NME):\n{}'.format(np_acctable))
            logging.info('Forgetting (NME): {}'.format(forgetting))


def _set_device(args):
    device_input = args["device"]
    if isinstance(device_input, list) and len(device_input) > 0 and isinstance(device_input[0], torch.device):
        return

    if isinstance(device_input, int):
        device = torch.device("cpu") if device_input == -1 else torch.device("cuda:{}".format(device_input))
        args["device"] = [device]
    elif isinstance(device_input, list):
        gpus = [torch.device("cpu") if d == -1 else torch.device("cuda:{}".format(d)) for d in device_input]
        args["device"] = gpus


def _set_random(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

    run_mode = args.get("run_mode")
    git_available = args.get("git_available")
    git_dirty = args.get("git_dirty")
    if run_mode == "debug" and git_available and git_dirty:
        logging.warning(
            "Running in debug mode with uncommitted changes. "
            "This run is not a clean, fully reproducible experiment snapshot."
        )


def _apply_data_root(args):
    data_root = args.get("data_root")
    if not data_root:
        return

    os.environ["TAGFEX_DATA_ROOT"] = data_root
    if "imagenet100" in str(args.get("dataset", "")).lower():
        os.environ["TAGFEX_IMAGENET100_ROOT"] = data_root


def _build_log_root(args, init_cls):
    base_log_dir = args.get("log_root", "logs")
    setting = str(args.get("setting") or args.get("data_protocol") or "").lower()
    if args.get("si_blurry", False) or setting in {"si_blurry", "flygcl"}:
        return os.path.join(
            base_log_dir,
            args["prefix"],
            args["dataset"],
            "si_blurry",
            "tasks{}_n{}_m{}".format(
                args.get("n_tasks", 5),
                args.get("n", 50),
                args.get("m", 10),
            ),
        )
    return os.path.join(
        base_log_dir,
        args["prefix"],
        args["dataset"],
        str(init_cls),
        str(args["increment"]),
    )


def _make_history_state(cnn_curve, nme_curve, cnn_matrix, nme_matrix):
    return {
        "cnn_curve": cnn_curve,
        "nme_curve": nme_curve,
        "cnn_matrix": cnn_matrix,
        "nme_matrix": nme_matrix,
    }


def _task_summary_payload(method, accy, include_old=True):
    if accy is None:
        return {}

    grouped = accy.get("grouped", {})
    payload = {
        f"Summary/{method}_total_acc": accy.get("top1"),
        f"Summary/{method}_new_acc": grouped.get("new"),
        f"Summary/{method}_top5": accy.get("top5"),
    }
    if include_old:
        payload[f"Summary/{method}_old_acc"] = grouped.get("old")
    return {key: value for key, value in payload.items() if value is not None}


def _runtime_reproducibility_metadata(args):
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tagfex_data_root": os.environ.get("TAGFEX_DATA_ROOT"),
        "local_rank": args.get("local_rank"),
        "is_distributed": args.get("is_distributed"),
        "run_mode": args.get("run_mode"),
        "config_path": args.get("config_path"),
        "run_command": args.get("run_command"),
        "git_branch": args.get("git_branch"),
        "git_commit": args.get("git_commit"),
        "git_dirty": args.get("git_dirty"),
    }


def _save_run_reproducibility(args, logs_name):
    os.makedirs(logs_name, exist_ok=True)
    resolved_config_path = os.path.join(logs_name, "resolved_config.json")
    run_metadata_path = os.path.join(logs_name, "run_reproducibility.json")
    args["resolved_config_path"] = resolved_config_path
    args["run_metadata_path"] = run_metadata_path

    with open(resolved_config_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(args), f, indent=2, ensure_ascii=False)

    with open(run_metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            _json_ready(
                {
                    "args": args,
                    "runtime": _runtime_reproducibility_metadata(args),
                    "checkpoint_note": "Task checkpoints include model weights, replay memory, class means, SIGReg/final-SIGReg states, and RNG states.",
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )


def _get_rng_state():
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    if not state:
        return
    try:
        if state.get("python_random_state") is not None:
            random.setstate(state["python_random_state"])
        if state.get("numpy_random_state") is not None:
            np.random.set_state(state["numpy_random_state"])
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"])
        cuda_state = state.get("torch_cuda_rng_state_all")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    except Exception as exc:
        logging.warning("Failed to restore checkpoint RNG state: {}".format(exc))


def _save_resume_checkpoint(model, save_path, task, history_state=None, extra_state=None):
    network = model._network.module if hasattr(model._network, "module") else model._network
    history_state = history_state or {}
    final_sigreg_state = (
        model.get_final_sigreg_state() if hasattr(model, "get_final_sigreg_state") else None
    )
    ckpt = {
        "task": task,
        "cur_task": model._cur_task,
        "known_classes": model._known_classes,
        "total_classes": model._total_classes,
        "model_state_dict": network.state_dict(),
        "args": copy.deepcopy(model.args),
        "args_json_ready": _json_ready(model.args),
        "runtime_reproducibility": _runtime_reproducibility_metadata(model.args),
        "rng_state": _get_rng_state(),
        "data_memory": model._data_memory,
        "targets_memory": model._targets_memory,
        "class_means": getattr(model, "_class_means", None),
        "sigreg_state": model.get_sigreg_state() if hasattr(model, "get_sigreg_state") else None,
        "final_sigreg_state": final_sigreg_state,
        "cnn_curve": history_state.get("cnn_curve", {"top1": [], "top5": []}),
        "nme_curve": history_state.get("nme_curve", {"top1": [], "top5": []}),
        "cnn_matrix": history_state.get("cnn_matrix", []),
        "nme_matrix": history_state.get("nme_matrix", []),
    }
    if extra_state:
        ckpt.update(extra_state)
    torch.save(ckpt, save_path)


def _eval_cnn_outputs(model):
    y_pred, y_true = model._eval_cnn(model.test_loader)
    return model._evaluate(y_pred, y_true), y_pred, y_true


def _eval_cnn_only(model):
    accy, _, _ = _eval_cnn_outputs(model)
    return accy


def _compute_analysis_class_means(model, data_manager):
    class_means = np.zeros((model._total_classes, model.feature_dim))
    batch_size = int(model.args.get("analysis_nme_batch_size", 64))
    num_workers = int(model.args.get("analysis_nme_num_workers", 4))
    pin_memory = getattr(model._device, "type", "cpu") == "cuda"

    for class_idx in range(model._total_classes):
        _, _, class_dataset = data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1),
            source="train",
            mode="test",
            ret_data=True,
        )
        class_loader = DataLoader(
            class_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        vectors, _ = model._extract_vectors(class_loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean_norm = np.linalg.norm(mean)
        if mean_norm > EPSILON:
            mean = mean / mean_norm
        class_means[class_idx, :] = mean

    return class_means


def _eval_nme_with_fresh_class_means(model, data_manager):
    class_means = _compute_analysis_class_means(model, data_manager)
    y_pred, y_true = model._eval_nme(model.test_loader, class_means)
    return model._evaluate(y_pred, y_true), y_pred, y_true, class_means


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _accuracy_row(stage, accy, method=None):
    row = {
        "stage": stage,
    }
    if method is not None:
        row["method"] = method
    if accy is None:
        return row

    row["top1"] = accy.get("top1")
    row["top5"] = accy.get("top5")
    for key, value in accy.get("grouped", {}).items():
        safe_key = str(key).replace("-", "_")
        row[f"group_{safe_key}"] = value
    return row


def _delta_row(before_row, after_row, method=None):
    row = {"stage": "delta_after_minus_before"}
    if method is not None:
        row["method"] = method
    for key, before_value in before_row.items():
        if key in ("stage", "method"):
            continue
        after_value = after_row.get(key)
        if isinstance(before_value, (int, float, np.number)) and isinstance(
            after_value, (int, float, np.number)
        ):
            row[key] = float(after_value) - float(before_value)
    return row


def _save_final_sigreg_accuracy(
    artifact_dir,
    task,
    before_cnn_accy,
    after_cnn_accy,
    before_nme_accy,
    after_nme_accy,
    final_state,
):
    os.makedirs(artifact_dir, exist_ok=True)
    before_cnn_row = _accuracy_row("before_final_sigreg", before_cnn_accy, "CNN")
    after_cnn_row = _accuracy_row("after_final_sigreg", after_cnn_accy, "CNN")
    cnn_delta_row = _delta_row(before_cnn_row, after_cnn_row, "CNN")
    rows = [before_cnn_row, after_cnn_row, cnn_delta_row]

    nme_delta_row = None
    if before_nme_accy is not None and after_nme_accy is not None:
        before_nme_row = _accuracy_row("before_final_sigreg", before_nme_accy, "NME_analysis")
        after_nme_row = _accuracy_row("after_final_sigreg", after_nme_accy, "NME_analysis")
        nme_delta_row = _delta_row(before_nme_row, after_nme_row, "NME_analysis")
        rows.extend([before_nme_row, after_nme_row, nme_delta_row])

    csv_path = os.path.join(artifact_dir, "final_sigreg_task_{}_accuracy.csv".format(task))
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(artifact_dir, "final_sigreg_task_{}_accuracy.json".format(task))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            _json_ready(
                {
                    "task_id": task,
                    "before_cnn": before_cnn_accy,
                    "after_cnn": after_cnn_accy,
                    "cnn_delta_after_minus_before": cnn_delta_row,
                    "before_nme_analysis": before_nme_accy,
                    "after_nme_analysis": after_nme_accy,
                    "nme_analysis_delta_after_minus_before": nme_delta_row,
                    "final_sigreg_state": final_state,
                }
            ),
            f,
            indent=2,
        )


def _save_model_confusions(model, artifact_dir, task, stage):
    cnn_accy, y_pred, y_true = _eval_cnn_outputs(model)
    _save_confusion_artifacts(
        artifact_dir,
        task,
        stage,
        "CNN",
        y_pred,
        y_true,
        model._total_classes,
        summary=cnn_accy,
    )

    class_means = getattr(model, "_class_means", None)
    if class_means is None:
        return

    y_pred, y_true = model._eval_nme(model.test_loader, class_means)
    nme_accy = model._evaluate(y_pred, y_true)
    _save_confusion_artifacts(
        artifact_dir,
        task,
        stage,
        "NME",
        y_pred,
        y_true,
        model._total_classes,
        summary=nme_accy,
    )


def _save_confusion_artifacts(
    artifact_dir,
    task,
    stage,
    method,
    y_pred,
    y_true,
    num_classes,
    summary=None,
):
    os.makedirs(artifact_dir, exist_ok=True)
    pred_top1 = y_pred.T[0] if getattr(y_pred, "ndim", 1) == 2 else y_pred
    raw = _confusion_matrix(y_true, pred_top1, num_classes)
    row_norm = _row_normalize_confusion(raw)

    safe_stage = str(stage).replace("/", "_")
    safe_method = str(method).replace("/", "_")
    stem = "task_{}_{}_{}".format(task, safe_stage, safe_method)

    raw_csv = os.path.join(artifact_dir, stem + "_confusion_raw.csv")
    norm_csv = os.path.join(artifact_dir, stem + "_confusion_row_normalized.csv")
    _write_confusion_csv(raw_csv, raw, integer=True)
    _write_confusion_csv(norm_csv, row_norm, integer=False)

    np.save(os.path.join(artifact_dir, stem + "_confusion_raw.npy"), raw)
    np.save(os.path.join(artifact_dir, stem + "_confusion_row_normalized.npy"), row_norm)

    meta_path = os.path.join(artifact_dir, stem + "_confusion_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            _json_ready(
                {
                    "task_id": task,
                    "stage": stage,
                    "method": method,
                    "num_classes": num_classes,
                    "summary": summary,
                    "raw_csv": raw_csv,
                    "row_normalized_csv": norm_csv,
                }
            ),
            f,
            indent=2,
        )

    _try_plot_confusion(
        raw,
        os.path.join(artifact_dir, stem + "_confusion_raw.png"),
        "{} {} task {} raw".format(stage, method, task),
        normalized=False,
    )
    _try_plot_confusion(
        row_norm,
        os.path.join(artifact_dir, stem + "_confusion_row_normalized.png"),
        "{} {} task {} row-normalized".format(stage, method, task),
        normalized=True,
    )


def _confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        true_label = int(true_label)
        pred_label = int(pred_label)
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix


def _row_normalize_confusion(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )


def _write_confusion_csv(path, matrix, integer=False):
    labels = list(range(matrix.shape[0]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for label, row in zip(labels, matrix):
            if integer:
                values = [int(v) for v in row]
            else:
                values = [float(np.around(v, decimals=6)) for v in row]
            writer.writerow([label] + values)


def _try_plot_confusion(matrix, path, title, normalized=False):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("matplotlib unavailable; skip confusion matrix figure {}: {}".format(path, exc))
        return

    num_classes = matrix.shape[0]
    fig_size = max(6, min(14, num_classes * 0.45))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.tick_params(axis="x", labelrotation=90)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if num_classes <= 20:
        threshold = matrix.max() * 0.5 if matrix.size and matrix.max() > 0 else 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                value = matrix[i, j]
                text = "{:.2f}".format(value) if normalized else str(int(value))
                color = "white" if value > threshold else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _resume_from_checkpoint(model, data_manager, resume_path):
    ckpt = torch.load(resume_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint {} is not a resumable task checkpoint.".format(resume_path))

    saved_task = ckpt["task"]
    total_classes = 0
    for task_id in range(saved_task + 1):
        total_classes += data_manager.get_task_size(task_id)
        model._network.update_fc(total_classes)

    model._network.load_state_dict(ckpt["model_state_dict"])
    model._cur_task = ckpt.get("cur_task", saved_task)
    model._known_classes = ckpt["known_classes"]
    model._total_classes = ckpt["total_classes"]
    model._data_memory = ckpt.get("data_memory", np.array([]))
    model._targets_memory = ckpt.get("targets_memory", np.array([]))

    class_means = ckpt.get("class_means", None)
    if class_means is not None:
        model._class_means = class_means

    sigreg_state = ckpt.get("sigreg_state", None)
    if hasattr(model, "load_sigreg_state"):
        if sigreg_state is not None:
            if model.load_sigreg_state(sigreg_state):
                logging.info("Loaded SIGReg matrix state from checkpoint.")
        else:
            logging.warning("Checkpoint does not contain SIGReg matrix state; SIGReg matrix will be reinitialized.")

    _restore_rng_state(ckpt.get("rng_state"))

    model.after_task()
    history_state = {
        "cnn_curve": ckpt.get("cnn_curve", {"top1": [], "top5": []}),
        "nme_curve": ckpt.get("nme_curve", {"top1": [], "top5": []}),
        "cnn_matrix": ckpt.get("cnn_matrix", []),
        "nme_matrix": ckpt.get("nme_matrix", []),
    }
    return saved_task + 1, history_state


def _resolve_resume_path(args, log_root):
    resume_path = args.get("resume")
    resume_dir = args.get("resume_dir")
    auto_resume = args.get("auto_resume", False)

    if resume_path:
        return resume_path

    if resume_dir:
        return _find_latest_checkpoint_in_dir(resume_dir)

    if auto_resume:
        if not os.path.exists(log_root):
            raise FileNotFoundError(
                "Auto resume requested, but log root does not exist: {}".format(log_root)
            )
        run_dirs = [
            os.path.join(log_root, d)
            for d in os.listdir(log_root)
            if os.path.isdir(os.path.join(log_root, d))
        ]
        if len(run_dirs) == 0:
            raise FileNotFoundError(
                "Auto resume requested, but no run directories were found under {}".format(log_root)
            )
        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        return _find_latest_checkpoint_in_dir(latest_run_dir)

    return None


def _find_latest_checkpoint_in_dir(run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError("Checkpoint directory not found: {}".format(ckpt_dir))

    ckpt_files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".pth")
        and "_task_" in f
        and f.rsplit("_task_", 1)[-1].split(".pth")[0].isdigit()
    ]
    if len(ckpt_files) == 0:
        raise FileNotFoundError("No task checkpoints found under {}".format(ckpt_dir))

    def _task_index(path):
        filename = os.path.basename(path)
        task_part = filename.rsplit("_task_", 1)[-1]
        task_str = task_part.split(".pth")[0]
        return int(task_str)

    return max(ckpt_files, key=_task_index)
