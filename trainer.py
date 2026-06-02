import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import time


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
    args["task_increments"] = data_manager.get_task_sizes()
    si_blurry_eval_groups = data_manager.get_si_blurry_eval_groups()
    if si_blurry_eval_groups is not None:
        args["si_blurry_eval_groups"] = si_blurry_eval_groups
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
        #accy是字典
        cnn_accy, nme_accy = model.eval_task()
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

            ckpt_dir = os.path.join(logs_name, "checkpoints")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            save_path = os.path.join(ckpt_dir, "{}_{}_task_{}.pth".format(
                args["prefix"], args["seed"], task
            ))

            history_state = {
                "cnn_curve": cnn_curve,
                "nme_curve": nme_curve,
                "cnn_matrix": cnn_matrix,
                "nme_matrix": nme_matrix,
            }
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


def _save_resume_checkpoint(model, save_path, task, history_state=None):
    network = model._network.module if hasattr(model._network, "module") else model._network
    history_state = history_state or {}
    ckpt = {
        "task": task,
        "cur_task": model._cur_task,
        "known_classes": model._known_classes,
        "total_classes": model._total_classes,
        "model_state_dict": network.state_dict(),
        "data_memory": model._data_memory,
        "targets_memory": model._targets_memory,
        "class_means": getattr(model, "_class_means", None),
        "sigreg_state": model.get_sigreg_state() if hasattr(model, "get_sigreg_state") else None,
        "cnn_curve": history_state.get("cnn_curve", {"top1": [], "top5": []}),
        "nme_curve": history_state.get("nme_curve", {"top1": [], "top5": []}),
        "cnn_matrix": history_state.get("cnn_matrix", []),
        "nme_matrix": history_state.get("nme_matrix", []),
    }
    torch.save(ckpt, save_path)


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
        if f.endswith(".pth") and "_task_" in f
    ]
    if len(ckpt_files) == 0:
        raise FileNotFoundError("No task checkpoints found under {}".format(ckpt_dir))

    def _task_index(path):
        filename = os.path.basename(path)
        task_part = filename.rsplit("_task_", 1)[-1]
        task_str = task_part.split(".pth")[0]
        return int(task_str)

    return max(ckpt_files, key=_task_index)
