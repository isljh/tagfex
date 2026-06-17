import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

from trainer import _set_device
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import accuracy, accuracy_by_groups


EPSILON = 1e-8


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_device(value):
    if value == "cpu":
        return -1
    if isinstance(value, str) and value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    return int(value)


def normalize_device_for_project(value):
    if isinstance(value, torch.device):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return parse_device(value)
    if isinstance(value, list):
        return [normalize_device_for_project(item) for item in value]
    return value


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    return value


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_args(config_path, checkpoint, device_override):
    args = load_json(config_path)
    ckpt_args = checkpoint.get("args_json_ready") or checkpoint.get("args") or {}
    merged = dict(args)
    merged.update(ckpt_args)

    if device_override is not None:
        merged["device"] = parse_device(device_override)
    else:
        merged["device"] = normalize_device_for_project(merged["device"])

    merged.setdefault("run_mode", "debug")
    merged.setdefault("is_distributed", False)
    # Avoid swanlab initialization in model constructors during offline analysis.
    merged["local_rank"] = 1

    seed = checkpoint.get("args", {}).get("seed", merged.get("seed", 0))
    if isinstance(seed, list):
        seed = seed[0]
    merged["seed"] = int(seed)

    _set_device(merged)
    return merged


def build_data_manager(args):
    init_cls_arg = args.get("init_cls", args.get("increment", 0))
    increment_arg = args.get("increment", init_cls_arg)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        init_cls_arg,
        increment_arg,
        args.get("aug", 1),
        args=args,
    )
    args["class_order"] = list(getattr(data_manager, "_class_order", []))
    args["task_increments"] = data_manager.get_task_sizes()
    si_blurry_eval_groups = data_manager.get_si_blurry_eval_groups()
    if si_blurry_eval_groups is not None:
        args["si_blurry_eval_groups"] = si_blurry_eval_groups
    return data_manager


def build_model(args, checkpoint, data_manager):
    model = factory.get_model(args["model_name"], args)
    saved_task = int(checkpoint["task"])
    total_classes = 0
    for task_id in range(saved_task + 1):
        total_classes += data_manager.get_task_size(task_id)
        model._network.update_fc(total_classes)

    model._network.load_state_dict(checkpoint["model_state_dict"])
    model._cur_task = int(checkpoint.get("cur_task", saved_task))
    model._known_classes = int(checkpoint["known_classes"])
    model._total_classes = int(checkpoint["total_classes"])
    model._network.to(model._device)
    model._network.eval()
    return model


def get_ptr(model):
    return model._network.module if hasattr(model._network, "module") else model._network


def extract_ta_features(model, loader, feature_kind):
    ptr = get_ptr(model)
    features, targets = [], []
    device = model._device

    with torch.no_grad():
        for batch in loader:
            inputs = batch[1].to(device)
            batch_targets = batch[-1]
            ta_model = ptr._get_active_ta_model() if hasattr(ptr, "_get_active_ta_model") else ptr.ta_net
            ta_fmap = ta_model(inputs)["fmaps"][-1]
            ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
            if feature_kind == "ta_feature":
                batch_features = ta_feature
            elif feature_kind == "embedding":
                batch_features = ptr.projector(ta_feature)
            else:
                raise ValueError("Unknown feature kind: {}".format(feature_kind))
            features.append(batch_features.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def l2_normalize(vectors):
    return (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T


def compute_oracle_class_means(vectors, targets, total_classes):
    vectors = l2_normalize(vectors)
    class_means = np.zeros((total_classes, vectors.shape[1]), dtype=np.float32)
    missing = []

    for class_idx in range(total_classes):
        class_vectors = vectors[targets == class_idx]
        if len(class_vectors) == 0:
            missing.append(class_idx)
            continue
        mean = np.mean(class_vectors, axis=0)
        norm = np.linalg.norm(mean)
        if norm > EPSILON:
            mean = mean / norm
        class_means[class_idx] = mean

    if missing:
        raise ValueError("No test samples found for classes: {}".format(missing))
    return class_means, vectors


def evaluate_nme(model, vectors, targets, class_means):
    dists = cdist(class_means, vectors, "sqeuclidean")
    y_pred = np.argsort(dists, axis=0)[: model.topk, :].T
    task_increments = model.args.get("task_increments")
    if task_increments is None:
        grouped = accuracy(y_pred.T[0], targets, model._known_classes)
    else:
        grouped = accuracy_by_groups(y_pred.T[0], targets, model._known_classes, task_increments)
    model._add_si_blurry_accuracy(grouped, y_pred.T[0], targets)
    top5 = np.around(
        (y_pred.T == np.tile(targets, (model.topk, 1))).sum() * 100 / len(targets),
        decimals=2,
    )
    return {
        "grouped": grouped,
        "top1": grouped["total"],
        "top{}".format(model.topk): top5,
    }, y_pred


def make_test_loader(model, data_manager, batch_size, num_workers):
    dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes),
        source="test",
        mode="test",
    )
    pin_memory = getattr(model._device, "type", "cpu") == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def save_outputs(output_dir, result, y_pred, y_true, class_means):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "oracle_nme_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, indent=2)
    np.save(os.path.join(output_dir, "oracle_nme_pred.npy"), y_pred)
    np.save(os.path.join(output_dir, "oracle_nme_target.npy"), y_true)
    np.save(os.path.join(output_dir, "oracle_nme_class_means.npy"), class_means)


def main():
    parser = argparse.ArgumentParser(
        description="Compute label-leaked test-prototype Oracle NME on TA features."
    )
    parser.add_argument("--config", required=True, help="Experiment config used by the checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Saved trainer checkpoint.")
    parser.add_argument(
        "--feature",
        choices=["ta_feature", "embedding"],
        default="ta_feature",
        help="Feature space used for oracle NME.",
    )
    parser.add_argument("--device", default=None, help="GPU id such as 0, or cpu.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    args_cli = parser.parse_args()

    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu")
    args = prepare_args(args_cli.config, checkpoint, args_cli.device)
    set_random(args["seed"])
    data_manager = build_data_manager(args)
    model = build_model(args, checkpoint, data_manager)

    if model._cur_task >= 0:
        data_manager.set_current_task(model._cur_task)

    loader = make_test_loader(model, data_manager, args_cli.batch_size, args_cli.num_workers)
    vectors, targets = extract_ta_features(model, loader, args_cli.feature)
    class_means, vectors = compute_oracle_class_means(vectors, targets, model._total_classes)
    result, y_pred = evaluate_nme(model, vectors, targets, class_means)

    summary = {
        "metric": "test_prototype_oracle_nme",
        "warning": "Label-leaked upper-bound diagnostic. Do not report as standard accuracy.",
        "config": args_cli.config,
        "checkpoint": args_cli.checkpoint,
        "feature": args_cli.feature,
        "task": model._cur_task,
        "total_classes": model._total_classes,
        "num_test_samples": int(len(targets)),
        "result": result,
    }
    print(json.dumps(json_ready(summary), indent=2))
    save_outputs(args_cli.output_dir, summary, y_pred, targets, class_means)


if __name__ == "__main__":
    main()

