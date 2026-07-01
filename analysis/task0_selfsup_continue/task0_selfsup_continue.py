
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab

from trainer import _apply_data_root, _set_device
from utils import factory
from utils.data_manager import DataManager


DEFAULT_CHECKPOINT = (
    "logs/ablation_lejepa_mean_fusion_online_probe/imagenet100_lejepa/0/10/"
    "20260625_181747/checkpoints/"
    "ablation_lejepa_mean_fusion_online_probe_1993_task_0.pth"
)


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


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_args(config_path, checkpoint, cli_args):
    args = load_json(config_path)
    ckpt_args = checkpoint.get("args_json_ready") or checkpoint.get("args") or {}
    merged = dict(args)
    merged.update(ckpt_args)

    if cli_args.device is not None:
        merged["device"] = [parse_device(cli_args.device)]
    merged.setdefault("device", [0])
    merged["is_distributed"] = False
    merged["local_rank"] = 1
    merged["online_linear_probe"] = True
    merged["probe_lr"] = cli_args.probe_lr
    merged["probe_weight_decay"] = cli_args.probe_weight_decay
    merged["probe_feature"] = cli_args.probe_feature
    merged["run_mode"] = "task0_selfsup_continue"
    merged["task0_selfsup_continue_from"] = str(cli_args.checkpoint)

    seed = checkpoint.get("args", {}).get("seed", merged.get("seed", 0))
    if isinstance(seed, list):
        seed = seed[0]
    merged["seed"] = int(seed)

    _set_device(merged)
    _apply_data_root(merged)
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
    return data_manager


def build_model(args, checkpoint, data_manager):
    model = factory.get_model(args["model_name"], args)
    saved_task = int(checkpoint["task"])
    if saved_task != 0:
        raise ValueError("Task0 continuation expects a task_0 checkpoint, got task {}.".format(saved_task))

    total_classes = data_manager.get_task_size(0)
    model._network.update_fc(total_classes)
    model._network.load_state_dict(checkpoint["model_state_dict"])
    model._cur_task = int(checkpoint.get("cur_task", saved_task))
    model._known_classes = int(checkpoint["known_classes"])
    model._total_classes = int(checkpoint["total_classes"])
    model._data_memory = checkpoint.get("data_memory", np.array([]))
    model._targets_memory = checkpoint.get("targets_memory", np.array([]))

    if hasattr(model, "load_sigreg_state"):
        model.load_sigreg_state(checkpoint.get("sigreg_state"),)
    if hasattr(model, "load_online_probe_state"):
        loaded_probe = model.load_online_probe_state(checkpoint.get("online_probe_state"))
    else:
        loaded_probe = False

    model._network.to(model._device)
    model.sig_reg.to(model._device)
    model._network.eval()
    return model, loaded_probe


def get_ptr(model):
    return model._network.module if hasattr(model._network, "module") else model._network


def make_task0_loader(model, data_manager, batch_size, num_workers):
    dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes),
        source="train",
        mode="train",
        appendent=None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=getattr(model._device, "type", "cpu") == "cuda",
        drop_last=True,
    )


def set_ta_projector_only_trainable(model):
    ptr = get_ptr(model)
    for param in ptr.parameters():
        param.requires_grad_(False)

    ta_model = ptr._get_active_ta_model()
    for param in ta_model.parameters():
        param.requires_grad_(True)
    for param in ptr.projector.parameters():
        param.requires_grad_(True)

    ptr.eval()
    ta_model.train()
    ptr.projector.train()
    return ptr, [p for p in list(ta_model.parameters()) + list(ptr.projector.parameters()) if p.requires_grad]


def extract_ta_and_embedding(ptr, inputs):
    ta_out = ptr._get_active_ta_model()(inputs)
    ta_fmap = ta_out["fmaps"][-1]
    ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
    embedding = ptr.projector(ta_feature)
    return ta_feature, embedding


def select_probe_feature(name, ta_feature, embedding):
    if name == "ta_feature":
        return ta_feature
    if name == "embedding":
        return embedding
    raise ValueError("Unknown probe_feature: {}".format(name))


def flatten_views(vs, targets, default_num_views):
    if vs.ndim == 5:
        num_items, num_views = vs.shape[:2]
        return vs.flatten(0, 1), targets.repeat_interleave(num_views), num_items, num_views
    if vs.ndim == 4:
        num_views = default_num_views
        if vs.shape[0] % num_views != 0:
            raise ValueError("Cannot infer num_items from flat input shape {} and num_views {}.".format(list(vs.shape), num_views))
        num_items = vs.shape[0] // num_views
        return vs, targets, num_items, num_views
    raise ValueError("Unexpected input shape: {}".format(list(vs.shape)))


def make_scheduler(optimizer, loader_len, epochs, lr):
    total_steps = max(1, loader_len * epochs)
    warmup_steps = min(loader_len, total_steps)
    if total_steps <= warmup_steps:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 1000)
    s1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=lr / 1000,
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])


def save_extra_checkpoint(model, args, checkpoint, output_dir, rows, cli_args, checkpoint_role="last", best_state=None, filename_suffix=None):
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lr_tag = "{:.0e}".format(cli_args.lr).replace("e-0", "e-").replace("e+0", "e")
    suffix = filename_suffix or "lr{}_ep{}".format(lr_tag, cli_args.epochs)
    save_path = ckpt_dir / "{}_{}_task_0_selfsup_continue_{}.pth".format(
        args["prefix"], args["seed"], suffix
    )
    ptr = get_ptr(model)
    extra_state = {
        "task": 0,
        "cur_task": model._cur_task,
        "known_classes": model._known_classes,
        "total_classes": model._total_classes,
        "model_state_dict": ptr.state_dict(),
        "args": args,
        "args_json_ready": json_ready(args),
        "source_checkpoint": str(cli_args.checkpoint),
        "source_checkpoint_has_online_probe_state": checkpoint.get("online_probe_state") is not None,
        "online_probe_state": model.get_online_probe_state() if hasattr(model, "get_online_probe_state") else None,
        "sigreg_state": model.get_sigreg_state() if hasattr(model, "get_sigreg_state") else None,
        "task0_selfsup_continue_state": {
            "checkpoint_role": checkpoint_role,
            "best_state": best_state,
            "epochs": cli_args.epochs,
            "lr": cli_args.lr,
            "weight_decay": cli_args.weight_decay,
            "batch_size": cli_args.batch_size,
            "num_workers": cli_args.num_workers,
            "loss_factor": cli_args.loss_factor,
            "num_rows": len(rows),
            "final_probe_top1": rows[-1].get("probe_top1") if rows else None,
            "final_probe_loss": rows[-1].get("probe_loss") if rows else None,
            "final_selfsup_loss": rows[-1].get("lejepa_loss") if rows else None,
            "probe_feature": cli_args.probe_feature,
        },
    }
    torch.save(extra_state, save_path)
    return save_path


def mean_metric(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def run(cli_args):
    checkpoint = torch.load(cli_args.checkpoint, map_location="cpu")
    args = prepare_args(cli_args.config, checkpoint, cli_args)
    set_random(args["seed"])
    data_manager = build_data_manager(args)
    model, loaded_probe = build_model(args, checkpoint, data_manager)
    loader = make_task0_loader(model, data_manager, cli_args.batch_size, cli_args.num_workers)
    ptr, trainable_params = set_ta_projector_only_trainable(model)
    if not trainable_params:
        raise RuntimeError("No trainable TA/projector parameters for Task0 self-supervised continuation.")

    optimizer = torch.optim.AdamW(trainable_params, lr=cli_args.lr, weight_decay=cli_args.weight_decay)
    scheduler = make_scheduler(optimizer, len(loader), cli_args.epochs, cli_args.lr)
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    swan_enabled = not cli_args.no_swanlab
    if swan_enabled:
        swanlab.init(
            project=cli_args.swanlab_project,
            experiment_name=cli_args.swanlab_name or "{}_task0_selfsup_continue".format(args["prefix"]),
            config=json_ready({
                "config": str(cli_args.config),
                "checkpoint": str(cli_args.checkpoint),
                "args": args,
                "loaded_probe_from_checkpoint": loaded_probe,
                "source_checkpoint_has_online_probe_state": checkpoint.get("online_probe_state") is not None,
                "epochs": cli_args.epochs,
                "lr": cli_args.lr,
                "weight_decay": cli_args.weight_decay,
                "probe_feature": cli_args.probe_feature,
            }),
            suffix="timestamp",
        )

    if not loaded_probe:
        print("Warning: source checkpoint has no online_probe_state; continuation probe starts from scratch.")

    rows = []
    epoch_rows = []
    best_state = {
        "metric": "epoch_mean_probe_top1",
        "mode": "max",
        "best_value": None,
        "best_epoch": None,
        "checkpoint": None,
    }
    global_step = 0
    lamb = args.get("lejepa_lambda", 0.05)
    default_num_views = int(args.get("num_views", 8))
    probe_feature_name = cli_args.probe_feature
    for epoch in range(cli_args.epochs):
        epoch_start = len(rows)
        ptr._get_active_ta_model().train()
        ptr.projector.train()
        prog = tqdm(loader, desc="Task0 selfsup continue epoch {}/{}".format(epoch + 1, cli_args.epochs), dynamic_ncols=True)
        for batch_idx, batch in enumerate(prog):
            vs = batch[1].to(model._device, non_blocking=True)
            targets = batch[-1].to(model._device, non_blocking=True)
            inputs, probe_targets, num_items, num_views = flatten_views(vs, targets, default_num_views)
            inputs = inputs.to(model._device, non_blocking=True)
            ta_feature, embedding = extract_ta_and_embedding(ptr, inputs)
            probe_features = select_probe_feature(probe_feature_name, ta_feature, embedding)
            inv_loss, sigreg_loss, lejepa_loss = model._compute_selfsup_losses(embedding, num_items, num_views, lamb)
            loss = lejepa_loss * cli_args.loss_factor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            probe_log = model._update_task0_online_probe(probe_features, probe_targets)
            row = {
                "epoch": epoch,
                "iter": batch_idx,
                "global_step": global_step,
                "loss": float(loss.detach().cpu()),
                "inv_loss": float(inv_loss.detach().cpu()),
                "sigreg_loss": float(sigreg_loss.detach().cpu()),
                "lejepa_loss": float(lejepa_loss.detach().cpu()),
                "probe_loss": probe_log.get("probe_loss"),
                "probe_top1": probe_log.get("probe_top1"),
                "lr": optimizer.param_groups[0]["lr"],
                "num_items": int(num_items),
                "num_views": int(num_views),
                "probe_feature": probe_feature_name,
            }
            rows.append(row)
            if swan_enabled:
                payload = {
                    "task0_extra/Prediction_Invariance_loss": row["inv_loss"],
                    "task0_extra/SIGReg_loss": row["sigreg_loss"],
                    "task0_extra/LeJEPA_total_loss": row["lejepa_loss"],
                    "task0_extra/lr": row["lr"],
                }
                if probe_log:
                    payload.update({
                        "task0_extra/probe_loss": probe_log["probe_loss"],
                        "task0_extra/probe_top1": probe_log["probe_top1"],
                    })
                swanlab.log(payload, step=global_step)
            prog.set_postfix({
                "loss": row["lejepa_loss"],
                "probe": row["probe_top1"],
                "lr": row["lr"],
            })
            global_step += 1

        current_epoch_rows = rows[epoch_start:]
        epoch_row = {
            "epoch": epoch,
            "global_step": max(global_step - 1, 0),
            "loss": mean_metric(current_epoch_rows, "loss"),
            "inv_loss": mean_metric(current_epoch_rows, "inv_loss"),
            "sigreg_loss": mean_metric(current_epoch_rows, "sigreg_loss"),
            "lejepa_loss": mean_metric(current_epoch_rows, "lejepa_loss"),
            "probe_loss": mean_metric(current_epoch_rows, "probe_loss"),
            "probe_top1": mean_metric(current_epoch_rows, "probe_top1"),
            "lr": current_epoch_rows[-1]["lr"] if current_epoch_rows else optimizer.param_groups[0]["lr"],
            "probe_feature": probe_feature_name,
        }
        epoch_rows.append(epoch_row)
        current_best_value = epoch_row.get("probe_top1")
        if current_best_value is not None and (
            best_state["best_value"] is None or current_best_value > best_state["best_value"]
        ):
            lr_tag = "{:.0e}".format(cli_args.lr).replace("e-0", "e-").replace("e+0", "e")
            best_state.update({
                "best_value": current_best_value,
                "best_epoch": epoch,
                "best_epoch_row": dict(epoch_row),
            })
            best_path = save_extra_checkpoint(
                model,
                args,
                checkpoint,
                output_dir,
                rows,
                cli_args,
                checkpoint_role="best_epoch_mean_probe",
                best_state=dict(best_state),
                filename_suffix="best_probe_lr{}_ep{}".format(lr_tag, cli_args.epochs),
            )
            best_state["checkpoint"] = str(best_path)
        if swan_enabled:
            payload = {
                "task0_extra_epoch/Prediction_Invariance_loss": epoch_row["inv_loss"],
                "task0_extra_epoch/SIGReg_loss": epoch_row["sigreg_loss"],
                "task0_extra_epoch/LeJEPA_total_loss": epoch_row["lejepa_loss"],
                "task0_extra_epoch/lr": epoch_row["lr"],
                "task0_extra_epoch/epoch": epoch + 1,
            }
            if epoch_row["probe_loss"] is not None:
                payload.update({
                    "task0_extra_epoch/probe_loss": epoch_row["probe_loss"],
                    "task0_extra_epoch/probe_top1": epoch_row["probe_top1"],
                })
            swanlab.log(payload, step=epoch_row["global_step"])

    csv_path = output_dir / "task0_selfsup_continue_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epoch", "iter", "global_step", "loss", "inv_loss", "sigreg_loss",
            "lejepa_loss", "probe_loss", "probe_top1", "lr", "num_items", "num_views", "probe_feature",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    epoch_csv_path = output_dir / "task0_selfsup_continue_epoch_metrics.csv"
    with epoch_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epoch", "global_step", "loss", "inv_loss", "sigreg_loss",
            "lejepa_loss", "probe_loss", "probe_top1", "lr", "probe_feature",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)

    save_path = save_extra_checkpoint(
        model,
        args,
        checkpoint,
        output_dir,
        rows,
        cli_args,
        checkpoint_role="last",
        best_state=dict(best_state),
    )
    summary_path = output_dir / "task0_selfsup_continue_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready({
            "checkpoint": str(cli_args.checkpoint),
            "config": str(cli_args.config),
            "output_dir": str(output_dir),
            "metrics_csv": str(csv_path),
            "epoch_metrics_csv": str(epoch_csv_path),
            "last_checkpoint": str(save_path),
            "best_checkpoint": best_state.get("checkpoint"),
            "best_state": best_state,
            "loaded_probe_from_checkpoint": loaded_probe,
            "source_checkpoint_has_online_probe_state": checkpoint.get("online_probe_state") is not None,
            "probe_feature": cli_args.probe_feature,
            "final_row": rows[-1] if rows else None,
            "final_epoch_row": epoch_rows[-1] if epoch_rows else None,
        }), f, indent=2)

    if swan_enabled:
        swanlab.finish()
    print("Saved metrics to {}".format(csv_path))
    print("Saved epoch metrics to {}".format(epoch_csv_path))
    print("Saved last continuation checkpoint to {}".format(save_path))
    if best_state.get("checkpoint"):
        print("Saved best continuation checkpoint to {}".format(best_state["checkpoint"]))


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run-config", default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = load_json(pre_args.run_config) if pre_args.run_config else {}

    parser = argparse.ArgumentParser(
        description="Task0-only LeJEPA self-supervised continuation from a saved checkpoint.",
        parents=[pre_parser],
    )
    parser.add_argument("--config", default=defaults.get("config", "exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_online_probe.json"))
    parser.add_argument("--checkpoint", default=defaults.get("checkpoint", DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=defaults.get("output_dir", "analysis/task0_selfsup_continue/results/ablation_lejepa_mean_fusion_online_probe_task0_extra"))
    parser.add_argument("--device", default=defaults.get("device", "0"))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 64))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 16))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 2e-4))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 5e-4))
    parser.add_argument("--loss-factor", type=float, default=defaults.get("loss_factor", 1.0))
    parser.add_argument("--probe-lr", type=float, default=defaults.get("probe_lr", 1e-3))
    parser.add_argument("--probe-weight-decay", type=float, default=defaults.get("probe_weight_decay", 1e-7))
    parser.add_argument("--probe-feature", choices=["ta_feature", "embedding"], default=defaults.get("probe_feature", "ta_feature"))
    parser.add_argument("--no-swanlab", action="store_true", default=defaults.get("no_swanlab", False))
    parser.add_argument("--swanlab-project", default=defaults.get("swanlab_project", "PyCIL_TagFex"))
    parser.add_argument("--swanlab-name", default=defaults.get("swanlab_name"))
    return parser.parse_args(remaining)


if __name__ == "__main__":
    run(parse_args())
