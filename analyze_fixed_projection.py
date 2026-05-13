import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_manager import DataManager
from utils.inc_net import TagFexNet


EPSILON = 1e-8
DEFAULT_HIST_DIRECTIONS = "0,1,2,3,4,5,10,20"
DEFAULT_CONFIG = "exps/tagfex_lejepa_mean_fusion_imagenet100_fixed_matrix.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze LeJEPA fixed random projection matrix Gaussianity."
    )
    parser.add_argument("--ckpt", required=True, help="Path to a resumable task checkpoint.")
    parser.add_argument(
        "--a-fixed",
        default=None,
        help="Path to A_fixed.pth. If omitted, tries to read sigreg_state.current.fixed_A from --ckpt.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Experiment JSON used to build the model/data.")
    parser.add_argument("--task-id", type=int, default=0, help="Task id to analyze.")
    parser.add_argument(
        "--ckpt-task-id",
        type=int,
        default=None,
        help="Task id represented by a raw state_dict checkpoint without a 'task' field.",
    )
    parser.add_argument("--data-root", default=None, help="Dataset root, e.g. path/to/ImageNet100.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV/figures.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or a GPU index.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--source", choices=["train", "test"], default="train")
    parser.add_argument("--mode", choices=["train", "test", "flip"], default="train")
    parser.add_argument("--num-random-eval", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Override the seed in --config.")
    parser.add_argument("--hist-directions", default=DEFAULT_HIST_DIRECTIONS)
    parser.add_argument("--projection-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Optional cap on flattened projector features. 0 means full task.",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-sigreg-loss", action="store_true")
    parser.add_argument("--pca", action="store_true", help="Also save PCA visualization of all_proj.")
    parser.add_argument("--max-pca-points", type=int, default=5000)
    return parser.parse_args()


def parse_device(device_arg):
    if device_arg is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    value = str(device_arg).lower()
    if value in {"cpu", "-1"}:
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda:0")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(device_arg)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_columns(matrix):
    return matrix / matrix.norm(p=2, dim=0, keepdim=True).clamp_min(EPSILON)


def configure_args(config, device, seed_override=None):
    config = dict(config)
    seed_value = config.get("seed", 0)
    if isinstance(seed_value, list):
        seed_value = seed_value[0]
    if seed_override is not None:
        seed_value = seed_override

    config["seed"] = int(seed_value)
    config["device"] = [device]
    config["is_distributed"] = False
    config["local_rank"] = 0
    config.setdefault("aug", 1)
    config.setdefault("run_mode", "debug")
    return config


def build_data_manager(config, data_root):
    if data_root:
        os.environ["TAGFEX_DATA_ROOT"] = data_root

    return DataManager(
        config["dataset"],
        config["shuffle"],
        config["seed"],
        config["init_cls"],
        config["increment"],
        config.get("aug", 1),
    )


def task_class_range(data_manager, task_id):
    if task_id < 0 or task_id >= data_manager.nb_tasks:
        raise ValueError(f"task-id {task_id} is outside [0, {data_manager.nb_tasks - 1}]")
    start = sum(data_manager.get_task_size(i) for i in range(task_id))
    end = start + data_manager.get_task_size(task_id)
    return start, end


def build_task_loader(data_manager, task_id, source, mode, batch_size, num_workers, device):
    start, end = task_class_range(data_manager, task_id)
    dataset = data_manager.get_dataset(
        np.arange(start, end),
        source=source,
        mode=mode,
        appendent=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    return loader, len(dataset), (start, end)


def checkpoint_state_and_task(ckpt_obj, fallback_task_id=None):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"], int(ckpt_obj.get("task", fallback_task_id or 0))
    if isinstance(ckpt_obj, dict):
        return ckpt_obj, int(fallback_task_id or 0)
    raise ValueError("Unsupported checkpoint format. Expected a dict or a resumable task checkpoint.")


def build_model(config, data_manager, checkpoint_task_id, state_dict, device):
    model = TagFexNet(config, False)
    total_classes = 0
    for task_id in range(checkpoint_task_id + 1):
        total_classes += data_manager.get_task_size(task_id)
        model.update_fc(total_classes)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not match TagFexNet. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )

    model.to(device)
    model.eval()
    return model


def find_matrix(obj, preferred_keys):
    if isinstance(obj, torch.Tensor) and obj.ndim == 2:
        return obj
    if not isinstance(obj, dict):
        return None

    for key in preferred_keys:
        value = obj.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value

    for key in ("sigreg_state", "current", "matrix_state"):
        value = obj.get(key)
        found = find_matrix(value, preferred_keys)
        if found is not None:
            return found

    task_states = obj.get("task_matrix_states")
    if isinstance(task_states, dict):
        for value in task_states.values():
            found = find_matrix(value, preferred_keys)
            if found is not None:
                return found

    return None


def reconstruct_fixed_matrix(config, feat_dim=1024, num_directions=256):
    seed = int(config.get("sigreg_matrix_seed", config.get("seed", 0)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(feat_dim, num_directions, generator=generator, dtype=torch.float32)
    return normalize_columns(matrix)


def load_fixed_matrix(a_fixed_path, ckpt_obj, config):
    preferred_keys = ("fixed_A", "A_fixed", "a_fixed", "A", "projection_matrix")
    if a_fixed_path:
        obj = torch_load(a_fixed_path, map_location="cpu")
        matrix = find_matrix(obj, preferred_keys)
        if matrix is None:
            raise ValueError(f"Could not find a 2D fixed matrix in {a_fixed_path}")
        return matrix.float()

    matrix = find_matrix(ckpt_obj, preferred_keys)
    if matrix is None:
        if config.get("sigreg_matrix_mode") == "fixed":
            print(
                "[warn] Could not find A_fixed in --ckpt; reconstructing fixed matrix "
                "from sigreg_matrix_seed/seed."
            )
            return reconstruct_fixed_matrix(config)
        raise ValueError("Could not find A_fixed in --ckpt; pass --a-fixed explicitly.")
    return matrix.float()


def validate_projection_setup(projector_dim, A_fixed):
    if projector_dim != 1024:
        raise ValueError(f"Expected projector output dim 1024, got {projector_dim}.")
    if tuple(A_fixed.shape) != (1024, 256):
        raise ValueError(f"Expected A_fixed shape [1024, 256], got {list(A_fixed.shape)}.")


def unpack_batch(batch):
    labels = batch[-1]
    if len(batch) == 3:
        inputs = batch[1]
    else:
        views = batch[1:-1]
        if not all(torch.is_tensor(v) for v in views):
            raise TypeError("Expected tensor views in batch[1:-1].")
        inputs = torch.stack(list(views), dim=1)
    return inputs, labels


def forward_projector(model, batch, device):
    inputs, labels = unpack_batch(batch)
    if inputs.ndim == 5:
        batch_size, num_views = inputs.shape[:2]
        flat_inputs = inputs.reshape(batch_size * num_views, *inputs.shape[2:])
        flat_labels = labels.repeat_interleave(num_views)
    elif inputs.ndim == 4:
        flat_inputs = inputs
        flat_labels = labels
    else:
        raise ValueError(f"Expected inputs with 4 or 5 dims, got shape {list(inputs.shape)}.")

    flat_inputs = flat_inputs.to(device, non_blocking=True)
    outputs = model(flat_inputs)
    if "embedding" not in outputs:
        raise KeyError("Model forward output does not contain 'embedding'.")
    proj = outputs["embedding"]
    return proj.detach().float().cpu(), flat_labels.detach().cpu()


def collect_projector_outputs(model, loader, device, max_features=0):
    proj_chunks = []
    label_chunks = []
    first_proj = None
    first_labels = None
    total = 0

    with torch.no_grad():
        for batch in loader:
            proj, labels = forward_projector(model, batch, device)
            if first_proj is None:
                first_proj = proj.clone()
                first_labels = labels.clone()

            if max_features and total + proj.size(0) > max_features:
                keep = max_features - total
                if keep <= 0:
                    break
                proj = proj[:keep]
                labels = labels[:keep]

            proj_chunks.append(proj)
            label_chunks.append(labels)
            total += proj.size(0)

            if max_features and total >= max_features:
                break

    if not proj_chunks:
        raise RuntimeError("No projector features were collected.")

    return (
        torch.cat(proj_chunks, dim=0),
        torch.cat(label_chunks, dim=0),
        first_proj,
        first_labels,
    )


def project_in_chunks(proj, matrix, device, chunk_size):
    chunks = []
    matrix_dev = matrix.to(device)
    with torch.no_grad():
        for start in range(0, proj.size(0), chunk_size):
            end = min(start + chunk_size, proj.size(0))
            z = proj[start:end].to(device, non_blocking=True).matmul(matrix_dev)
            chunks.append(z.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def compute_direction_stats(z):
    z = z.float()
    mean = z.mean(dim=0)
    centered = z - mean
    variance = centered.square().mean(dim=0)
    std = variance.sqrt()
    safe_std = std.clamp_min(EPSILON)
    skewness = centered.pow(3).mean(dim=0) / safe_std.pow(3)
    kurtosis = centered.pow(4).mean(dim=0) / safe_std.pow(4)

    mean_abs = mean.abs()
    std_abs_error = (std - 1.0).abs()
    skew_abs = skewness.abs()
    kurtosis_abs_error = (kurtosis - 3.0).abs()

    per_direction = []
    for direction in range(z.size(1)):
        per_direction.append(
            {
                "direction": direction,
                "mean": float(mean[direction]),
                "std": float(std[direction]),
                "skewness": float(skewness[direction]),
                "kurtosis": float(kurtosis[direction]),
                "mean_abs": float(mean_abs[direction]),
                "std_abs_error": float(std_abs_error[direction]),
                "skew_abs": float(skew_abs[direction]),
                "kurtosis_abs_error": float(kurtosis_abs_error[direction]),
            }
        )

    summary = {
        "num_samples": int(z.size(0)),
        "num_directions": int(z.size(1)),
        "mean_abs_avg": float(mean_abs.mean()),
        "mean_abs_max": float(mean_abs.max()),
        "std_mean": float(std.mean()),
        "std_var": float(std.var(unbiased=False)),
        "std_abs_error_avg": float(std_abs_error.mean()),
        "skew_abs_avg": float(skew_abs.mean()),
        "kurtosis_mean": float(kurtosis.mean()),
        "kurtosis_abs_error_avg": float(kurtosis_abs_error.mean()),
    }
    return {"summary": summary, "per_direction": per_direction}


def sigreg_loss_from_projection(z, knots=17, chunk_size=16384):
    device = z.device
    t = torch.linspace(0, 3, knots, dtype=torch.float32, device=device)
    dt = 3 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, dtype=torch.float32, device=device)
    weights[[0, -1]] = dt
    phi = torch.exp(-t.square() / 2.0)
    weights = weights * phi

    cos_sum = torch.zeros(z.size(1), knots, dtype=torch.float64, device=device)
    sin_sum = torch.zeros(z.size(1), knots, dtype=torch.float64, device=device)
    total = 0
    with torch.no_grad():
        for start in range(0, z.size(0), chunk_size):
            end = min(start + chunk_size, z.size(0))
            x_t = z[start:end].float().unsqueeze(-1) * t
            cos_sum += x_t.cos().sum(dim=0).double()
            sin_sum += x_t.sin().sum(dim=0).double()
            total += end - start

    cos_mean = (cos_sum / total).float()
    sin_mean = (sin_sum / total).float()
    err = (cos_mean - phi).square() + sin_mean.square()
    statistic = (err @ weights) * total
    return float(statistic.mean())


def save_stats(name, stats, output_dir):
    dump_json(stats, output_dir / f"{name}_stats.json")
    csv_path = output_dir / f"{name}_stats.csv"
    fieldnames = [
        "direction",
        "mean",
        "std",
        "skewness",
        "kurtosis",
        "mean_abs",
        "std_abs_error",
        "skew_abs",
        "kurtosis_abs_error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats["per_direction"])


def write_rows_csv(rows, path):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_hist_directions(value):
    directions = []
    for item in value.split(","):
        item = item.strip()
        if item:
            directions.append(int(item))
    return directions


def get_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping figures: {exc}")
        return None


def plot_histograms(z, directions, prefix, fig_dir, plt):
    if plt is None:
        return
    z_np = z.numpy()
    for direction in directions:
        if direction < 0 or direction >= z_np.shape[1]:
            continue
        values = z_np[:, direction]
        left = min(float(np.min(values)), -4.0)
        right = max(float(np.max(values)), 4.0)
        xs = np.linspace(left, right, 500)
        normal_pdf = np.exp(-0.5 * xs * xs) / math.sqrt(2 * math.pi)

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=60, density=True, alpha=0.65, color="#4C78A8", label="projected")
        plt.plot(xs, normal_pdf, color="#D62728", linewidth=2, label="N(0, 1)")
        plt.title(f"{prefix} direction {direction}")
        plt.xlabel("projection value")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_hist_dir_{direction}.png", dpi=160)
        plt.close()


def plot_metric_per_direction(stats, prefix, fig_dir, plt):
    if plt is None:
        return
    metrics = [
        ("mean", "mean"),
        ("std", "std"),
        ("skewness", "skew"),
        ("kurtosis", "kurtosis"),
    ]
    directions = [row["direction"] for row in stats["per_direction"]]
    for metric, suffix in metrics:
        values = [row[metric] for row in stats["per_direction"]]
        plt.figure(figsize=(8, 4))
        plt.plot(directions, values, linewidth=1.5)
        if metric == "std":
            plt.axhline(1.0, color="#D62728", linestyle="--", linewidth=1)
        elif metric == "kurtosis":
            plt.axhline(3.0, color="#D62728", linestyle="--", linewidth=1)
        elif metric in {"mean", "skewness"}:
            plt.axhline(0.0, color="#D62728", linestyle="--", linewidth=1)
        plt.xlabel("direction")
        plt.ylabel(metric)
        plt.title(f"{prefix} {metric} per direction")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_{suffix}_per_direction.png", dpi=160)
        plt.close()


def plot_pca(all_proj, labels, fig_dir, max_points, seed, plt):
    if plt is None or all_proj.size(0) < 2:
        return

    num_points = min(max_points, all_proj.size(0))
    generator = torch.Generator().manual_seed(seed)
    if num_points < all_proj.size(0):
        indices = torch.randperm(all_proj.size(0), generator=generator)[:num_points]
        x = all_proj[indices].float()
        y = labels[indices].numpy()
    else:
        x = all_proj.float()
        y = labels.numpy()

    x = x - x.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(x, q=2, center=False)
        coords = x @ v[:, :2]
    except Exception:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        coords = x @ vh[:2].T

    coords_np = coords.numpy()
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c=y,
        s=5,
        alpha=0.65,
        cmap="tab20",
        linewidths=0,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("task feature PCA")
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(fig_dir / "task0_pca_features.png", dpi=180)
    plt.close()


def generate_random_matrix(feat_dim, num_directions, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(feat_dim, num_directions, generator=generator, dtype=torch.float32)
    return normalize_columns(matrix)


def random_multi_summary(random_stats):
    metrics = [
        "mean_abs_avg",
        "std_abs_error_avg",
        "skew_abs_avg",
        "kurtosis_abs_error_avg",
    ]
    summary = {
        "num_random_eval": len(random_stats),
        "num_samples": random_stats[0]["summary"]["num_samples"],
        "num_directions": random_stats[0]["summary"]["num_directions"],
    }
    for metric in metrics:
        values = np.array([stats["summary"][metric] for stats in random_stats], dtype=np.float64)
        summary[f"random_eval_{metric}_mean"] = float(values.mean())
        summary[f"random_eval_{metric}_std"] = float(values.std(ddof=0))
    return summary


def add_sigreg(stats, z, skip, chunk_size):
    if not skip:
        stats["summary"]["global_sigreg_loss_optional"] = sigreg_loss_from_projection(
            z, chunk_size=chunk_size
        )
    else:
        stats["summary"]["global_sigreg_loss_optional"] = None


def summary_row(name, stats):
    row = {"name": name}
    for key in (
        "num_samples",
        "num_directions",
        "mean_abs_avg",
        "std_abs_error_avg",
        "skew_abs_avg",
        "kurtosis_abs_error_avg",
        "global_sigreg_loss_optional",
    ):
        row[key] = stats["summary"].get(key)
    return row


def random_average_row(random_stats):
    metrics = [
        "mean_abs_avg",
        "std_abs_error_avg",
        "skew_abs_avg",
        "kurtosis_abs_error_avg",
        "global_sigreg_loss_optional",
    ]
    row = {
        "name": "task_random_avg",
        "num_samples": random_stats[0]["summary"]["num_samples"],
        "num_directions": random_stats[0]["summary"]["num_directions"],
    }
    for metric in metrics:
        values = [stats["summary"].get(metric) for stats in random_stats]
        values = [v for v in values if v is not None]
        row[metric] = float(np.mean(values)) if values else None
    return row


def main():
    args = parse_args()
    if args.num_random_eval < 1:
        raise ValueError("--num-random-eval must be at least 1.")

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = parse_device(args.device)
    config = configure_args(load_json(args.config), device, args.seed)
    set_seed(config["seed"])

    ckpt_obj = torch_load(args.ckpt, map_location="cpu")
    state_dict, checkpoint_task_id = checkpoint_state_and_task(ckpt_obj, args.ckpt_task_id)

    data_manager = build_data_manager(config, args.data_root)
    loader, raw_dataset_size, class_range = build_task_loader(
        data_manager,
        args.task_id,
        args.source,
        args.mode,
        args.batch_size,
        args.num_workers,
        device,
    )
    model = build_model(config, data_manager, checkpoint_task_id, state_dict, device)

    A_fixed = load_fixed_matrix(args.a_fixed, ckpt_obj, config)
    A_fixed = normalize_columns(A_fixed)

    all_proj, all_labels, batch_proj, batch_labels = collect_projector_outputs(
        model, loader, device, max_features=args.max_features
    )
    projector_dim = int(all_proj.size(1))
    validate_projection_setup(projector_dim, A_fixed)

    hist_directions = parse_hist_directions(args.hist_directions)
    plt = None if args.skip_figures else get_pyplot()

    batch_z_fixed = project_in_chunks(batch_proj, A_fixed, device, args.projection_chunk_size)
    batch_fixed_stats = compute_direction_stats(batch_z_fixed)
    add_sigreg(batch_fixed_stats, batch_z_fixed, args.skip_sigreg_loss, args.projection_chunk_size)
    save_stats("batch_fixed", batch_fixed_stats, output_dir)

    task_z_fixed = project_in_chunks(all_proj, A_fixed, device, args.projection_chunk_size)
    task_fixed_stats = compute_direction_stats(task_z_fixed)
    add_sigreg(task_fixed_stats, task_z_fixed, args.skip_sigreg_loss, args.projection_chunk_size)
    save_stats("task_fixed", task_fixed_stats, output_dir)

    random_stats = []
    first_random_z = None
    for eval_id in range(args.num_random_eval):
        A_random = generate_random_matrix(projector_dim, 256, args.random_seed + eval_id)
        z_random = project_in_chunks(all_proj, A_random, device, args.projection_chunk_size)
        stats = compute_direction_stats(z_random)
        add_sigreg(stats, z_random, args.skip_sigreg_loss, args.projection_chunk_size)
        stats["summary"]["random_eval_id"] = eval_id
        stats["summary"]["random_seed"] = args.random_seed + eval_id
        random_stats.append(stats)
        if eval_id == 0:
            first_random_z = z_random
            save_stats("task_random", stats, output_dir)

    multi_summary = random_multi_summary(random_stats)
    multi_eval = {
        "summary": multi_summary,
        "per_eval": [stats["summary"] for stats in random_stats],
    }
    dump_json(multi_eval, output_dir / "task_random_multi_eval_summary.json")
    write_rows_csv([multi_summary], output_dir / "task_random_multi_eval_summary.csv")

    summary_rows = [
        summary_row("batch_fixed", batch_fixed_stats),
        summary_row("task_fixed", task_fixed_stats),
        random_average_row(random_stats),
    ]
    dump_json(
        {
            "metadata": {
                "checkpoint": args.ckpt,
                "a_fixed": args.a_fixed or "checkpoint:sigreg_state.current.fixed_A",
                "config": args.config,
                "task_id": args.task_id,
                "checkpoint_task_id": checkpoint_task_id,
                "source": args.source,
                "mode": args.mode,
                "class_range": list(class_range),
                "raw_dataset_size": raw_dataset_size,
                "projector_output_dim": projector_dim,
                "a_fixed_shape": list(A_fixed.shape),
                "num_random_eval": args.num_random_eval,
            },
            "rows": summary_rows,
        },
        output_dir / "analysis_summary.json",
    )
    write_rows_csv(summary_rows, output_dir / "analysis_summary.csv")

    plot_histograms(batch_z_fixed, hist_directions, "batch_fixed", fig_dir, plt)
    plot_histograms(task_z_fixed, hist_directions, "task_fixed", fig_dir, plt)
    plot_histograms(first_random_z, hist_directions, "task_random", fig_dir, plt)
    plot_metric_per_direction(task_fixed_stats, "task_fixed", fig_dir, plt)
    plot_metric_per_direction(random_stats[0], "task_random", fig_dir, plt)
    if args.pca:
        plot_pca(all_proj, all_labels, fig_dir, args.max_pca_points, config["seed"], plt)

    print(f"Collected projector features: {list(all_proj.shape)}")
    print(f"Batch fixed summary: {batch_fixed_stats['summary']}")
    print(f"Task fixed summary: {task_fixed_stats['summary']}")
    print(f"Random multi-eval summary: {multi_summary}")
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
