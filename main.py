print("开始 import")
import json
print("json OK")
import argparse
print("argparse OK")
import os
print("os OK")
import torch
print("torch OK")
import torch.distributed as dist
print("dist OK")
from trainer import train
print("trainer OK")

def main():
    print("开始执行 main.py")
    # 1. 初始化分布式环境
    # DDP 启动时会自动注入环境变量 LOCAL_RANK, RANK, WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # 判断是否为分布式启动
    is_distributed = local_rank != -1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        device_id = local_rank
        torch.cuda.set_device(device_id)
    else:
        # 非 DDP 模式保留原逻辑或默认 0
        device_id = 0

    args = setup_parser().parse_args()   # 拿命令行里的参数
    param = load_json(args.config)     # 拿 JSON 文件里的参数
    args_dict = vars(args)

    # 2. 合并参数
    for key in ['dataset', 'init_cls', 'increment']:
        if args_dict.get(key) is not None:
            param[key] = args_dict[key]

    # 3. 强制更新设备信息为当前进程分配的显卡
    # DDP 模式下，param['device'] 应该传给 trainer 当前进程的 ID
    param['device'] = device_id
    param['is_distributed'] = is_distributed
    param['local_rank'] = local_rank

    args_dict.update(param)

    # 4. 只有主进程打印日志，避免重复
    if local_rank <= 0:
        print(f"Distributed training: {is_distributed}, Current device: {device_id}")

    train(args_dict)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='PyCIL DDP Version')
    parser.add_argument('--config', type=str, default='./exps/finetune.json')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--init_cls', type=int)
    parser.add_argument('--increment', type=int)
    # 注意：DDP 模式下 --device 应当被弃用，由 torchrun 控制
    parser.add_argument('--device', type=str, help='Deprecated in DDP mode')
    return parser


if __name__ == '__main__':
    main()