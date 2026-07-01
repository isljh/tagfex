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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab

from convs.linears import TagFex_SimpleLinear
from models.tagfex_lejepa import SIGReg
from trainer import _apply_data_root, _set_device
from utils.data_manager import DataManager
from utils.inc_net import get_convnet


DEFAULT_BASE_CONFIG = "exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_online_probe.json"


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


def get_rng_state():
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if state.get("torch_cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


class StandaloneTALEJEPA(nn.Module):
    def __init__(self, args):
        super().__init__()
        conv_args = dict(args)
        conv_args["convnet_type"] = args.get("ta_convnet_type", args.get("convnet_type", "resnet18"))
        self.encoder = get_convnet(conv_args)
        self.ta_feature_dim = self.encoder.out_dim
        hidden_dim = int(args["proj_hidden_dim"])
        output_dim = int(args["proj_output_dim"])
        self.projector = nn.Sequential(
            TagFex_SimpleLinear(self.ta_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            TagFex_SimpleLinear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            TagFex_SimpleLinear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        out = self.encoder(x)
        fmap = out["fmaps"][-1]
        ta_feature = fmap.flatten(2).permute(0, 2, 1).mean(1)
        embedding = self.projector(ta_feature)
        return {"ta_feature": ta_feature, "embedding": embedding}


def make_probe(feature_dim, num_classes, norm):
    layers = []
    if norm == "layernorm":
        layers.append(nn.LayerNorm(feature_dim))
    elif norm == "batchnorm":
        layers.append(nn.BatchNorm1d(feature_dim))
    elif norm in ("none", None):
        pass
    else:
        raise ValueError("Unknown probe_norm: {}".format(norm))
    layers.append(nn.Linear(feature_dim, num_classes))
    return nn.Sequential(*layers)


def prepare_args(cli_args):
    args = load_json(cli_args.base_config)
    args["device"] = [parse_device(cli_args.device)] if cli_args.device is not None else args.get("device", [0])
    args["is_distributed"] = False
    args["local_rank"] = 1
    args["run_mode"] = "standalone_ta_lejepa_probe"
    args["num_views"] = cli_args.num_views
    args["lejepa_lambda"] = cli_args.lejepa_lambda
    args["probe_norm"] = cli_args.probe_norm
    seed = args.get("seed", 0)
    if isinstance(seed, list):
        seed = seed[0]
    args["seed"] = int(seed)
    _set_device(args)
    _apply_data_root(args)
    return args


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


def make_task0_loaders(args, data_manager, batch_size, eval_batch_size, num_workers):
    num_classes = data_manager.get_task_size(0)
    task0_classes = np.arange(0, num_classes)
    train_dataset = data_manager.get_dataset(
        task0_classes,
        source="train",
        mode="train",
        appendent=None,
    )
    train_eval_dataset = data_manager.get_dataset(
        task0_classes,
        source="train",
        mode="test",
        appendent=None,
    )
    test_eval_dataset = data_manager.get_dataset(
        task0_classes,
        source="test",
        mode="test",
        appendent=None,
    )
    device = args["device"][0]
    pin_memory = isinstance(device, torch.device) and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_eval_loader = DataLoader(
        test_eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, train_eval_loader, test_eval_loader, num_classes


def flatten_views(vs, targets, default_num_views):
    if vs.ndim == 5:
        num_items, num_views = vs.shape[:2]
        return vs.flatten(0, 1), targets.repeat_interleave(num_views), num_items, num_views
    if vs.ndim == 4:
        num_views = default_num_views
        if vs.shape[0] % num_views != 0:
            raise ValueError("Cannot infer batch size from flat input shape {} and num_views {}.".format(list(vs.shape), num_views))
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


def compute_selfsup_losses(sigreg, embedding, num_items, num_views, lamb, sigreg_view_mode):
    proj = embedding.reshape(num_items, num_views, -1)
    proj_mean = proj.mean(1, keepdim=True)
    inv_loss = (proj_mean - proj).square().mean()
    if sigreg_view_mode == "mixed":
        sigreg_loss = sigreg(embedding)
    elif sigreg_view_mode == "view_wise":
        losses = [sigreg(proj[:, view_idx, :]) for view_idx in range(num_views)]
        sigreg_loss = torch.stack(losses).mean()
    else:
        raise ValueError("Unknown sigreg_view_mode: {}".format(sigreg_view_mode))
    lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
    return inv_loss, sigreg_loss, lejepa_loss


def update_probe(probe, optimizer, features, targets):
    probe_features = features.detach().float()
    if probe_features.size(0) != targets.size(0):
        raise ValueError(
            "Probe feature/target batch mismatch: {} vs {}.".format(probe_features.size(0), targets.size(0))
        )
    logits = probe(probe_features)
    loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = logits.argmax(dim=1)
    top1 = pred.eq(targets).float().mean().item() * 100.0
    return float(loss.detach().cpu()), top1


def evaluate_probe(model, probe, loader, device):
    was_training = model.training
    probe_was_training = probe.training
    model.eval()
    probe.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch[1].to(device, non_blocking=True)
            targets = batch[-1].to(device, non_blocking=True)
            out = model(inputs)
            logits = probe(out["ta_feature"].float())
            loss = F.cross_entropy(logits, targets)
            pred = logits.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
            total_loss += float(loss.detach().cpu()) * targets.size(0)
    if was_training:
        model.train()
    if probe_was_training:
        probe.train()
    if total == 0:
        return {"loss": None, "top1": None}
    return {"loss": total_loss / total, "top1": correct * 100.0 / total}


def mean_metric(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def save_checkpoint(
    path,
    model,
    probe,
    sigreg,
    args,
    rows,
    epoch_rows,
    cli_args,
    checkpoint_role="last",
    best_state=None,
    optimizer=None,
    probe_optimizer=None,
    scheduler=None,
    current_epoch=None,
    global_step=0,
    start_epoch=0,
    end_epoch=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "projector_state_dict": model.projector.state_dict(),
            "probe_state_dict": probe.state_dict(),
            "sigreg_state": sigreg.get_matrix_state() if hasattr(sigreg, "get_matrix_state") else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "probe_optimizer_state_dict": probe_optimizer.state_dict() if probe_optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": get_rng_state(),
            "args": args,
            "args_json_ready": json_ready(args),
            "standalone_ta_lejepa_state": {
                "checkpoint_role": checkpoint_role,
                "best_state": best_state,
                "current_epoch": current_epoch,
                "global_step": global_step,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "epochs": cli_args.epochs,
                "extra_epochs": cli_args.extra_epochs,
                "resume_checkpoint": str(cli_args.resume_checkpoint) if cli_args.resume_checkpoint else None,
                "lr": cli_args.lr,
                "weight_decay": cli_args.weight_decay,
                "batch_size": cli_args.batch_size,
                "eval_batch_size": cli_args.eval_batch_size,
                "eval_every": cli_args.eval_every,
                "eval_train": cli_args.eval_train,
                "num_workers": cli_args.num_workers,
                "lejepa_lambda": cli_args.lejepa_lambda,
                "sigreg_view_mode": cli_args.sigreg_view_mode,
                "probe_norm": cli_args.probe_norm,
                "probe_lr": cli_args.probe_lr,
                "probe_weight_decay": cli_args.probe_weight_decay,
                "num_rows": len(rows),
                "final_row": rows[-1] if rows else None,
                "final_epoch_row": epoch_rows[-1] if epoch_rows else None,
            },
        },
        path,
    )


def load_resume_checkpoint(path, model, probe, optimizer, probe_optimizer, device, reset_lr, lr, probe_lr):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    probe.load_state_dict(ckpt["probe_state_dict"])
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("probe_optimizer_state_dict") is not None:
        probe_optimizer.load_state_dict(ckpt["probe_optimizer_state_dict"])
    if reset_lr:
        for group in optimizer.param_groups:
            group["lr"] = lr
        for group in probe_optimizer.param_groups:
            group["lr"] = probe_lr
    restore_rng_state(ckpt.get("rng_state"))
    state = ckpt.get("standalone_ta_lejepa_state", {})
    current_epoch = state.get("current_epoch")
    start_epoch = int(current_epoch) + 1 if current_epoch is not None else 0
    global_step = int(state.get("global_step", 0))
    best_state = state.get("best_state") or {
        "metric": "test_probe_top1",
        "mode": "max",
        "best_value": None,
        "best_epoch": None,
        "checkpoint": None,
    }
    return start_epoch, global_step, best_state


def resolve_epoch_range(cli_args, resume_start_epoch):
    if cli_args.extra_epochs is not None:
        if cli_args.extra_epochs <= 0:
            raise ValueError("--extra-epochs must be positive when provided.")
        return resume_start_epoch, resume_start_epoch + cli_args.extra_epochs
    if cli_args.epochs < resume_start_epoch or (cli_args.resume_checkpoint and cli_args.epochs <= resume_start_epoch):
        raise ValueError(
            "Target --epochs ({}) must be greater than resumed start epoch ({}). Use --extra-epochs to append training.".format(
                cli_args.epochs, resume_start_epoch
            )
        )
    return resume_start_epoch, cli_args.epochs


def run(cli_args):
    args = prepare_args(cli_args)
    set_random(args["seed"])
    data_manager = build_data_manager(args)
    loader, train_eval_loader, test_eval_loader, num_classes = make_task0_loaders(
        args, data_manager, cli_args.batch_size, cli_args.eval_batch_size, cli_args.num_workers
    )
    device = args["device"][0]

    model = StandaloneTALEJEPA(args).to(device)
    probe = make_probe(model.ta_feature_dim, num_classes, cli_args.probe_norm).to(device)
    sigreg = SIGReg(matrix_mode=args.get("sigreg_matrix_mode", "per_batch_random")).to(device)
    if hasattr(sigreg, "set_matrix_seed"):
        sigreg.set_matrix_seed(args.get("seed", 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
    probe_optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=cli_args.probe_lr,
        weight_decay=cli_args.probe_weight_decay,
    )
    resume_start_epoch = 0
    global_step = 0
    best_state = {
        "metric": "test_probe_top1",
        "mode": "max",
        "best_value": None,
        "best_epoch": None,
        "checkpoint": None,
    }
    if cli_args.resume_checkpoint:
        resume_start_epoch, global_step, best_state = load_resume_checkpoint(
            cli_args.resume_checkpoint,
            model,
            probe,
            optimizer,
            probe_optimizer,
            device,
            cli_args.resume_reset_lr,
            cli_args.lr,
            cli_args.probe_lr,
        )
    start_epoch, end_epoch = resolve_epoch_range(cli_args, resume_start_epoch)
    run_epochs = end_epoch - start_epoch
    scheduler = make_scheduler(optimizer, len(loader), run_epochs, optimizer.param_groups[0]["lr"])

    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    swan_enabled = not cli_args.no_swanlab
    if swan_enabled:
        swanlab.init(
            project=cli_args.swanlab_project,
            experiment_name=cli_args.swanlab_name or "standalone_ta_lejepa_probe",
            config=json_ready({
                "base_config": str(cli_args.base_config),
                "args": args,
                "epochs": cli_args.epochs,
                "extra_epochs": cli_args.extra_epochs,
                "resume_checkpoint": str(cli_args.resume_checkpoint) if cli_args.resume_checkpoint else None,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "lr": cli_args.lr,
                "weight_decay": cli_args.weight_decay,
                "batch_size": cli_args.batch_size,
                "eval_batch_size": cli_args.eval_batch_size,
                "eval_every": cli_args.eval_every,
                "eval_train": cli_args.eval_train,
                "num_views": cli_args.num_views,
                "lejepa_lambda": cli_args.lejepa_lambda,
                "sigreg_view_mode": cli_args.sigreg_view_mode,
                "probe_norm": cli_args.probe_norm,
            }),
            suffix="timestamp",
        )

    rows = []
    epoch_rows = []
    prog_bar = tqdm(range(start_epoch, end_epoch), desc="Standalone TA-LeJEPA", dynamic_ncols=True)
    for epoch in prog_bar:
        epoch_start = len(rows)
        model.train()
        probe.train()
        for batch_idx, batch in enumerate(loader):
            vs = batch[1].to(device, non_blocking=True)
            targets = batch[-1].to(device, non_blocking=True)
            inputs, probe_targets, num_items, num_views = flatten_views(vs, targets, cli_args.num_views)
            inputs = inputs.to(device, non_blocking=True)

            out = model(inputs)
            ta_feature = out["ta_feature"]
            embedding = out["embedding"]
            inv_loss, sigreg_loss, lejepa_loss = compute_selfsup_losses(
                sigreg, embedding, num_items, num_views, cli_args.lejepa_lambda, cli_args.sigreg_view_mode
            )

            optimizer.zero_grad()
            lejepa_loss.backward()
            optimizer.step()
            scheduler.step()

            probe_loss, probe_top1 = update_probe(probe, probe_optimizer, ta_feature, probe_targets)
            row = {
                "epoch": epoch,
                "iter": batch_idx,
                "global_step": global_step,
                "inv_loss": float(inv_loss.detach().cpu()),
                "sigreg_loss": float(sigreg_loss.detach().cpu()),
                "lejepa_loss": float(lejepa_loss.detach().cpu()),
                "probe_loss": probe_loss,
                "probe_top1": probe_top1,
                "lr": optimizer.param_groups[0]["lr"],
                "num_items": int(num_items),
                "num_views": int(num_views),
            }
            rows.append(row)
            if swan_enabled:
                swanlab.log(
                    {
                        "standalone_ta/Prediction_Invariance_loss": row["inv_loss"],
                        "standalone_ta/SIGReg_loss": row["sigreg_loss"],
                        "standalone_ta/LeJEPA_total_loss": row["lejepa_loss"],
                        "standalone_ta/probe_loss": row["probe_loss"],
                        "standalone_ta/probe_top1": row["probe_top1"],
                        "standalone_ta/lr": row["lr"],
                    },
                    step=global_step,
                )
            global_step += 1

        current_epoch_rows = rows[epoch_start:]
        should_eval = cli_args.eval_every > 0 and (((epoch + 1) % cli_args.eval_every == 0) or epoch == end_epoch - 1)
        if should_eval and cli_args.eval_train:
            train_eval = evaluate_probe(model, probe, train_eval_loader, device)
        else:
            train_eval = {"loss": None, "top1": None}
        if should_eval:
            test_eval = evaluate_probe(model, probe, test_eval_loader, device)
        else:
            test_eval = {"loss": None, "top1": None}
        epoch_row = {
            "epoch": epoch,
            "global_step": max(global_step - 1, 0),
            "inv_loss": mean_metric(current_epoch_rows, "inv_loss"),
            "sigreg_loss": mean_metric(current_epoch_rows, "sigreg_loss"),
            "lejepa_loss": mean_metric(current_epoch_rows, "lejepa_loss"),
            "probe_loss": mean_metric(current_epoch_rows, "probe_loss"),
            "probe_top1": mean_metric(current_epoch_rows, "probe_top1"),
            "train_probe_loss": train_eval["loss"],
            "train_probe_top1": train_eval["top1"],
            "test_probe_loss": test_eval["loss"],
            "test_probe_top1": test_eval["top1"],
            "lr": current_epoch_rows[-1]["lr"] if current_epoch_rows else optimizer.param_groups[0]["lr"],
        }
        epoch_rows.append(epoch_row)
        current_best_value = epoch_row.get("test_probe_top1")
        if current_best_value is not None and (
            best_state["best_value"] is None or current_best_value > best_state["best_value"]
        ):
            best_state.update({
                "best_value": current_best_value,
                "best_epoch": epoch,
                "best_epoch_row": dict(epoch_row),
            })
            best_path = output_dir / "checkpoints" / "standalone_ta_lejepa_seed{}_task0_best_test_probe.pth".format(args["seed"])
            best_state["checkpoint"] = str(best_path)
            save_checkpoint(
                best_path,
                model,
                probe,
                sigreg,
                args,
                rows,
                epoch_rows,
                cli_args,
                "best_test_probe",
                dict(best_state),
                optimizer,
                probe_optimizer,
                scheduler,
                epoch,
                global_step,
                start_epoch,
                end_epoch,
            )
        postfix = {
            "loss": epoch_row["lejepa_loss"],
            "probe": epoch_row["probe_top1"],
            "lr": epoch_row["lr"],
        }
        if epoch_row["test_probe_top1"] is not None:
            postfix["test"] = epoch_row["test_probe_top1"]
        prog_bar.set_postfix(postfix)
        if swan_enabled:
            payload = {
                "standalone_ta_epoch/Prediction_Invariance_loss": epoch_row["inv_loss"],
                "standalone_ta_epoch/SIGReg_loss": epoch_row["sigreg_loss"],
                "standalone_ta_epoch/LeJEPA_total_loss": epoch_row["lejepa_loss"],
                "standalone_ta_epoch/probe_loss": epoch_row["probe_loss"],
                "standalone_ta_epoch/probe_top1": epoch_row["probe_top1"],
                "standalone_ta_epoch/lr": epoch_row["lr"],
                "standalone_ta_epoch/epoch": epoch + 1,
            }
            if epoch_row["train_probe_top1"] is not None:
                payload.update({
                    "standalone_ta_eval/train_probe_loss": epoch_row["train_probe_loss"],
                    "standalone_ta_eval/train_probe_top1": epoch_row["train_probe_top1"],
                })
            if epoch_row["test_probe_top1"] is not None:
                payload.update({
                    "standalone_ta_eval/test_probe_loss": epoch_row["test_probe_loss"],
                    "standalone_ta_eval/test_probe_top1": epoch_row["test_probe_top1"],
                })
            swanlab.log(payload, step=epoch_row["global_step"])

    metrics_csv = output_dir / "standalone_ta_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epoch", "iter", "global_step", "inv_loss", "sigreg_loss", "lejepa_loss",
            "probe_loss", "probe_top1", "lr", "num_items", "num_views",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    epoch_csv = output_dir / "standalone_ta_epoch_metrics.csv"
    with epoch_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epoch", "global_step", "inv_loss", "sigreg_loss", "lejepa_loss",
            "probe_loss", "probe_top1", "train_probe_loss", "train_probe_top1",
            "test_probe_loss", "test_probe_top1", "lr",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)

    ckpt_name = "standalone_ta_lejepa_seed{}_task0_last_ep{}.pth".format(args["seed"], end_epoch)
    checkpoint_path = output_dir / "checkpoints" / ckpt_name
    save_checkpoint(
        checkpoint_path,
        model,
        probe,
        sigreg,
        args,
        rows,
        epoch_rows,
        cli_args,
        "last",
        dict(best_state),
        optimizer,
        probe_optimizer,
        scheduler,
        end_epoch - 1 if end_epoch > start_epoch else None,
        global_step,
        start_epoch,
        end_epoch,
    )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            json_ready({
                "base_config": str(cli_args.base_config),
                "output_dir": str(output_dir),
                "resume_checkpoint": str(cli_args.resume_checkpoint) if cli_args.resume_checkpoint else None,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "extra_epochs": cli_args.extra_epochs,
                "metrics_csv": str(metrics_csv),
                "epoch_metrics_csv": str(epoch_csv),
                "last_checkpoint": str(checkpoint_path),
                "best_checkpoint": best_state.get("checkpoint"),
                "best_state": best_state,
                "num_classes": num_classes,
                "feature_dim": model.ta_feature_dim,
                "final_row": rows[-1] if rows else None,
                "final_epoch_row": epoch_rows[-1] if epoch_rows else None,
            }),
            f,
            indent=2,
        )

    if swan_enabled:
        swanlab.finish()
    print("Saved metrics to {}".format(metrics_csv))
    print("Saved epoch metrics to {}".format(epoch_csv))
    print("Saved last checkpoint to {}".format(checkpoint_path))
    if best_state.get("checkpoint"):
        print("Saved best checkpoint to {}".format(best_state["checkpoint"]))


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run-config", default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = load_json(pre_args.run_config) if pre_args.run_config else {}

    parser = argparse.ArgumentParser(description="Standalone Task0 TA-LeJEPA training with online linear probe.", parents=[pre_parser])
    parser.add_argument("--base-config", default=defaults.get("base_config", DEFAULT_BASE_CONFIG))
    parser.add_argument("--output-dir", default=defaults.get("output_dir", "standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200"))
    parser.add_argument("--device", default=defaults.get("device", "0"))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 200))
    parser.add_argument("--extra-epochs", type=int, default=defaults.get("extra_epochs"))
    parser.add_argument("--resume-checkpoint", default=defaults.get("resume_checkpoint"))
    parser.add_argument("--resume-reset-lr", action=argparse.BooleanOptionalAction, default=defaults.get("resume_reset_lr", True))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 64))
    parser.add_argument("--eval-batch-size", type=int, default=defaults.get("eval_batch_size", 256))
    parser.add_argument("--eval-every", type=int, default=defaults.get("eval_every", 10))
    parser.add_argument("--eval-train", action=argparse.BooleanOptionalAction, default=defaults.get("eval_train", False))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 16))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 5e-4))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 5e-4))
    parser.add_argument("--lejepa-lambda", type=float, default=defaults.get("lejepa_lambda", 0.05))
    parser.add_argument("--sigreg-view-mode", choices=["mixed", "view_wise"], default=defaults.get("sigreg_view_mode", "mixed"))
    parser.add_argument("--probe-lr", type=float, default=defaults.get("probe_lr", 1e-3))
    parser.add_argument("--probe-weight-decay", type=float, default=defaults.get("probe_weight_decay", 1e-7))
    parser.add_argument("--probe-norm", choices=["layernorm", "batchnorm", "none"], default=defaults.get("probe_norm", "layernorm"))
    parser.add_argument("--num-views", type=int, default=defaults.get("num_views", 8))
    parser.add_argument("--no-swanlab", action="store_true", default=defaults.get("no_swanlab", False))
    parser.add_argument("--swanlab-project", default=defaults.get("swanlab_project", "PyCIL_TagFex"))
    parser.add_argument("--swanlab-name", default=defaults.get("swanlab_name", "ta_lejepa_probe_task0_lr5e-4_ep200"))
    return parser.parse_args(remaining)


if __name__ == "__main__":
    run(parse_args())
