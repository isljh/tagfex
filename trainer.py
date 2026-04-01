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

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    # 1. 只有主进程创建文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logs_name = "logs/{}/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'], timestamp)
    if local_rank <= 0:
        if not os.path.exists(logs_name):
            os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"],timestamp,
        args["prefix"], args["seed"], args["convnet_type"],
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
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["aug"] if "aug" in args else 1,
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    # 6. 开始任务循环
    for task in range(data_manager.nb_tasks):
        if local_rank <= 0:
            logging.info("All params: {}".format(count_parameters(model._network)))
            logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        # 增量训练
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        # ======= 在这里插入强制保存权重的逻辑 =======
        if local_rank <= 0:
            # 自动生成权重保存路径
            ckpt_dir = os.path.join(logs_name, "checkpoints")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            save_path = os.path.join(ckpt_dir, "{}_{}_task_{}.pth".format(
                args["prefix"], args["seed"], task
            ))

            # 执行保存
            torch.save(model._network.state_dict(), save_path)
            logging.info("!!! Model checkpoint saved to: {} !!!".format(save_path))
        # =========================================

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