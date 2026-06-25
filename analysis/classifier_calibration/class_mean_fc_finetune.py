import argparse
import copy
import csv
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from trainer import _set_device
from utils import factory
from utils.data_manager import DataManager


EPSILON = 1e-8


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.device):
        return str(value)
    return value


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
    model._data_memory = checkpoint.get("data_memory", np.array([]))
    model._targets_memory = checkpoint.get("targets_memory", np.array([]))
    model._network.to(model._device)
    model._network.eval()
    return model


def get_ptr(model):
    return model._network.module if hasattr(model._network, "module") else model._network


def make_seen_test_loader(model, data_manager, batch_size, num_workers):
    dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes),
        source="test",
        mode="test",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=getattr(model._device, "type", "cpu") == "cuda",
    )


def make_finetune_loader(model, data_manager, checkpoint, batch_size, num_workers, feature_mode):
    task_id = int(checkpoint["task"])
    task_start = sum(data_manager.get_task_size(i) for i in range(task_id))
    task_end = model._total_classes

    memory_data = checkpoint.get("data_memory", np.array([]))
    memory_targets = checkpoint.get("targets_memory", np.array([]))
    if len(memory_targets) > 0:
        old_mask = memory_targets < task_start
        old_memory = (memory_data[old_mask], memory_targets[old_mask])
        if len(old_memory[1]) == 0:
            old_memory = None
    else:
        old_memory = None

    dataset = data_manager.get_dataset(
        np.arange(task_start, task_end),
        source="train",
        mode=feature_mode,
        appendent=old_memory,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=getattr(model._device, "type", "cpu") == "cuda",
        drop_last=False,
    ), task_start, task_end


def repeat_targets_for_inputs(inputs, targets):
    if inputs.ndim == 5:
        return inputs.flatten(0, 1), targets.repeat_interleave(inputs.shape[1])
    return inputs, targets


def extract_features(model, loader):
    model._network.eval()
    features, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract concat features", dynamic_ncols=True):
            inputs = batch[1].to(model._device, non_blocking=True)
            batch_targets = batch[-1].to(model._device, non_blocking=True)
            inputs, batch_targets = repeat_targets_for_inputs(inputs, batch_targets)
            outputs = model._network(inputs)
            features.append(outputs["features"].detach().cpu())
            targets.append(batch_targets.detach().cpu())
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def evaluate_cnn(model, loader, desc="Evaluate CNN"):
    model._network.eval()
    y_pred, y_true = [], []
    for _, (_, inputs, targets) in enumerate(tqdm(loader, desc=desc, dynamic_ncols=True)):
        inputs = inputs.to(model._device)
        with torch.no_grad():
            outputs = model._network(inputs)["logits"]
        predicts = torch.topk(outputs, k=model.topk, dim=1, largest=True, sorted=True)[1]
        y_pred.append(predicts.cpu().numpy())
        y_true.append(targets.cpu().numpy())
    return model._evaluate(np.concatenate(y_pred), np.concatenate(y_true))


def freeze_except_fc(model):
    ptr = get_ptr(model)
    for param in ptr.parameters():
        param.requires_grad_(False)
    for param in ptr.fc.parameters():
        param.requires_grad_(True)
    ptr.fc.train()


def reset_fc_like_initialization(fc):
    fc.reset_parameters()


def l2_normalize_tensor(x):
    return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(EPSILON)


def compute_class_mean_weights(features, targets, num_classes, scale_mode, reference_weight):
    norm_features = l2_normalize_tensor(features.float())
    class_means = torch.zeros(num_classes, features.shape[1], dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.long)
    missing = []

    for class_idx in range(num_classes):
        mask = targets == class_idx
        counts[class_idx] = int(mask.sum())
        if counts[class_idx] == 0:
            missing.append(class_idx)
            continue
        mean = norm_features[mask].mean(dim=0)
        class_means[class_idx] = mean / mean.norm(p=2).clamp_min(EPSILON)

    if missing:
        raise ValueError("No finetune samples for classes: {}".format(missing))

    if scale_mode == "none":
        return class_means, counts
    if scale_mode == "global_weight_norm":
        scale = reference_weight.detach().float().norm(p=2, dim=1).mean()
        return class_means * scale, counts
    if scale_mode == "per_class_weight_norm":
        scales = reference_weight.detach().float().norm(p=2, dim=1).clamp_min(EPSILON)
        return class_means * scales[:, None], counts
    raise ValueError("Unknown class mean scale mode: {}".format(scale_mode))


def train_fc_on_features(model, features, targets, epochs, lr, weight_decay, batch_size):
    ptr = get_ptr(model)
    device = model._device
    dataset = TensorDataset(features.float(), targets.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(ptr.fc.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    epoch_bar = tqdm(range(epochs), desc="Finetune fc", dynamic_ncols=True)
    for _ in epoch_bar:
        total_loss, total_items = 0.0, 0
        ptr.fc.train()
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            logits = ptr.fc(batch_features)
            loss = torch.nn.functional.cross_entropy(logits, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_targets.size(0)
            total_items += batch_targets.size(0)
        avg_loss = total_loss / max(1, total_items)
        losses.append(avg_loss)
        epoch_bar.set_postfix(loss="{:.4f}".format(avg_loss))
    return losses


def clone_fc_state(model):
    ptr = get_ptr(model)
    return copy.deepcopy(ptr.fc.state_dict())


def load_fc_state(model, state):
    get_ptr(model).fc.load_state_dict(state)


def run_branch(model, base_fc_state, branch, features, targets, test_loader, args_cli):
    load_fc_state(model, base_fc_state)
    ptr = get_ptr(model)

    if branch == "original":
        return {
            "branch": branch,
            "init": "checkpoint_fc",
            "losses": [],
            "accuracy": evaluate_cnn(model, test_loader, desc="Evaluate original"),
        }

    freeze_except_fc(model)
    if branch == "random":
        reset_fc_like_initialization(ptr.fc)
        init_name = "random_reinit"
        counts = None
    elif branch == "class_mean":
        reference_weight = ptr.fc.weight.detach().cpu().clone()
        weights, counts = compute_class_mean_weights(
            features,
            targets,
            model._total_classes,
            args_cli.class_mean_scale,
            reference_weight,
        )
        with torch.no_grad():
            ptr.fc.weight.copy_(weights.to(model._device))
            if ptr.fc.bias is not None:
                ptr.fc.bias.zero_()
        init_name = "class_mean"
    else:
        raise ValueError("Unknown branch: {}".format(branch))

    losses = train_fc_on_features(
        model,
        features,
        targets,
        args_cli.epochs,
        args_cli.lr,
        args_cli.weight_decay,
        args_cli.fc_batch_size,
    )
    return {
        "branch": branch,
        "init": init_name,
        "losses": losses,
        "class_counts": counts.tolist() if counts is not None else None,
        "accuracy": evaluate_cnn(model, test_loader, desc="Evaluate {}".format(branch)),
    }


def accuracy_row(result):
    accy = result["accuracy"]
    row = {
        "branch": result["branch"],
        "init": result["init"],
        "top1": accy.get("top1"),
        "top5": accy.get("top5"),
        "final_loss": result["losses"][-1] if result["losses"] else None,
    }
    for key, value in accy.get("grouped", {}).items():
        row["group_{}".format(str(key).replace("-", "_"))] = value
    return row


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.4f}".format(value) if abs(value) < 1 else "{:.2f}".format(value)
    return str(value)


def make_markdown_table(rows):
    preferred = [
        "branch",
        "init",
        "top1",
        "top5",
        "group_total",
        "group_00_09",
        "group_10_19",
        "final_loss",
    ]
    columns = [column for column in preferred if any(column in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [format_value(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def print_result_table(summary):
    rows = [accuracy_row(result) for result in summary["results"]]
    print("\nResult table:")
    print(make_markdown_table(rows))


def save_summary(output_dir, summary):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "class_mean_fc_finetune_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, indent=2)

    rows = [accuracy_row(result) for result in summary["results"]]
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(os.path.join(output_dir, "class_mean_fc_finetune_accuracy.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(output_dir, "class_mean_fc_finetune_accuracy.md"), "w", encoding="utf-8") as f:
        f.write(make_markdown_table(rows))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Classifier-only finetune diagnostic with random vs class-mean fc initialization."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-mode", default="test", choices=["train", "test"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--fc-batch-size", type=int, default=256)
    parser.add_argument(
        "--class-mean-scale",
        default="global_weight_norm",
        choices=["none", "global_weight_norm", "per_class_weight_norm"],
    )
    parser.add_argument("--output-dir", default=None)
    args_cli = parser.parse_args()

    print("[1/6] Loading checkpoint and config...")
    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu")
    args = prepare_args(args_cli.config, checkpoint, args_cli.device)
    set_random(args["seed"])

    print("[2/6] Building data manager...")
    data_manager = build_data_manager(args)
    data_manager.set_current_task(int(checkpoint["task"]))
    print("[3/6] Rebuilding model and loading weights...")
    model = build_model(args, checkpoint, data_manager)

    if int(checkpoint["task"]) <= 0:
        raise ValueError("This diagnostic is intended for Task1+ checkpoints, not Task0.")

    print("[4/6] Building eval and finetune loaders...")
    test_loader = make_seen_test_loader(model, data_manager, args_cli.batch_size, args_cli.num_workers)
    finetune_loader, task_start, task_end = make_finetune_loader(
        model,
        data_manager,
        checkpoint,
        args_cli.batch_size,
        args_cli.num_workers,
        args_cli.feature_mode,
    )
    print("[5/6] Extracting cached concat features for fc-only finetune...")
    features, targets = extract_features(model, finetune_loader)
    print("Cached features: {} targets: {}".format(tuple(features.shape), tuple(targets.shape)))
    base_fc_state = clone_fc_state(model)

    print("[6/6] Running branches: original, random, class_mean...")
    results = []
    for branch in ["original", "random", "class_mean"]:
        print("Running branch: {}".format(branch))
        results.append(run_branch(model, base_fc_state, branch, features, targets, test_loader, args_cli))

    summary = {
        "analysis": "class_mean_fc_classifier_only_finetune",
        "config": args_cli.config,
        "checkpoint": args_cli.checkpoint,
        "task": int(checkpoint["task"]),
        "task_class_range": [int(task_start), int(task_end)],
        "total_classes": int(model._total_classes),
        "feature_dim": int(features.shape[1]),
        "num_finetune_features": int(features.shape[0]),
        "feature_mode": args_cli.feature_mode,
        "epochs": args_cli.epochs,
        "lr": args_cli.lr,
        "weight_decay": args_cli.weight_decay,
        "fc_batch_size": args_cli.fc_batch_size,
        "class_mean_scale": args_cli.class_mean_scale,
        "results": results,
    }
    print_result_table(summary)
    print("\nFull JSON summary:")
    print(json.dumps(json_ready(summary), indent=2))
    save_summary(args_cli.output_dir, summary)


if __name__ == "__main__":
    main()
