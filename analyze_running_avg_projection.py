import argparse
import shlex
import sys
from pathlib import Path

import torch

from analyze_fixed_projection import (
    DEFAULT_HIST_DIRECTIONS,
    DEFAULT_NUM_DIRECTIONS,
    add_sigreg,
    build_data_manager,
    build_model,
    build_task_loader,
    checkpoint_state_and_task,
    collect_projector_outputs,
    compute_direction_stats,
    configure_args,
    dump_json,
    generate_random_matrix,
    get_pyplot,
    load_json,
    normalize_columns,
    parse_device,
    parse_hist_directions,
    plot_histograms,
    plot_metric_per_direction,
    plot_pca,
    project_in_chunks,
    random_average_row,
    random_multi_summary,
    resolve_checkpoint_path,
    save_stats,
    set_seed,
    summary_row,
    torch_load,
    validate_projection_setup,
    write_rows_csv,
)


DEFAULT_CONFIG = "exps/tagfex_lejepa_mean_fusion_imagenet100_running_avg_matrix.json"
DEFAULT_OUTPUT_DIR = "outputs/running_avg_projection_analysis"
PROJECTION_METRICS = [
    "mean_abs_avg",
    "std_abs_error_avg",
    "skew_abs_avg",
    "kurtosis_abs_error_avg",
    "global_sigreg_loss_optional",
]
GEOMETRY_LOWER_IS_BETTER = {
    "column_norm_std",
    "cosine_offdiag_abs_mean",
    "cosine_offdiag_abs_std",
    "cosine_offdiag_abs_p95",
    "cosine_offdiag_abs_p99",
    "cosine_offdiag_abs_max",
    "condition_number",
}
GEOMETRY_HIGHER_IS_BETTER = {
    "effective_rank_entropy",
    "effective_rank_energy",
    "stable_rank",
    "singular_value_min",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze saved LeJEPA SIGReg running-average projection matrix Gaussianity."
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Path to a resumable task checkpoint. If omitted, resolves the newest "
            "matching checkpoint from --config log_root/prefix/dataset/init_cls/increment."
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Experiment JSON used to build the model/data.")
    parser.add_argument("--log-root", default=None, help="Override config log_root when auto-resolving --ckpt.")
    parser.add_argument("--run-id", default=None, help="Specific run timestamp, e.g. 20260506_112117.")
    parser.add_argument("--task-id", type=int, default=0, help="Target data task id to analyze.")
    parser.add_argument(
        "--matrix-task-id",
        type=int,
        default=None,
        help=(
            "Saved running_A task id to use. If omitted, uses sigreg_state.current.running_A. "
            "If set, prefers sigreg_state.task_matrix_states[str(id)].running_A."
        ),
    )
    parser.add_argument(
        "--ckpt-task-id",
        type=int,
        default=None,
        help="Task id represented by a raw state_dict checkpoint without a 'task' field.",
    )
    parser.add_argument("--data-root", default=None, help="Dataset root, e.g. /root/autodl-tmp/datasets/ImageNet100.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON/CSV/figures.")
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
    parser.add_argument("--num-projection-directions", type=int, default=DEFAULT_NUM_DIRECTIONS)
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
    parser.add_argument(
        "--inspect-matrix-only",
        action="store_true",
        help="Only load checkpoint and print saved running_A metadata; do not load data/model.",
    )
    return parser.parse_args()


def _matrix_metadata(state, source):
    matrix = state.get("running_A")
    return {
        "matrix_source": source,
        "matrix_mode": state.get("matrix_mode"),
        "matrix_task_id": state.get("task_id"),
        "running_count": state.get("running_count"),
        "matrix_seed": state.get("matrix_seed"),
        "running_A_shape": list(matrix.shape) if isinstance(matrix, torch.Tensor) else None,
    }


def _validate_running_state(state, source):
    if not isinstance(state, dict):
        raise ValueError(f"{source} is not a SIGReg matrix state dict.")
    if state.get("matrix_mode") != "running_avg":
        raise ValueError(f"{source} matrix_mode is {state.get('matrix_mode')!r}, expected 'running_avg'.")
    matrix = state.get("running_A")
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise ValueError(f"{source} does not contain a 2D running_A tensor.")


def load_running_matrix(ckpt_obj, matrix_task_id=None):
    sigreg_state = ckpt_obj.get("sigreg_state") if isinstance(ckpt_obj, dict) else None
    if not isinstance(sigreg_state, dict):
        raise ValueError(
            "Checkpoint does not contain sigreg_state. "
            "This analysis requires a checkpoint saved with running_A."
        )

    current = sigreg_state.get("current")
    task_states = sigreg_state.get("task_matrix_states") or {}

    if matrix_task_id is not None:
        key = str(matrix_task_id)
        if key in task_states:
            state = task_states[key]
            source = f"checkpoint:sigreg_state.task_matrix_states[{key}].running_A"
        elif isinstance(current, dict) and int(current.get("task_id", -1)) == int(matrix_task_id):
            state = current
            source = "checkpoint:sigreg_state.current.running_A"
        else:
            available = sorted(task_states.keys())
            current_task = current.get("task_id") if isinstance(current, dict) else None
            raise ValueError(
                f"Could not find running_A for matrix-task-id {matrix_task_id}. "
                f"Available task_matrix_states={available}, current_task_id={current_task}."
            )
    else:
        state = current
        source = "checkpoint:sigreg_state.current.running_A"

    _validate_running_state(state, source)
    matrix = state["running_A"].float()
    metadata = _matrix_metadata(state, source)
    metadata["available_task_matrix_states"] = sorted(task_states.keys())
    return matrix, metadata


def inspect_matrix(ckpt_path, ckpt_obj, matrix_task_id):
    matrix, metadata = load_running_matrix(ckpt_obj, matrix_task_id)
    print(f"Checkpoint: {ckpt_path}")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print(f"running_A dtype: {matrix.dtype}")


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    if value is None:
        return None
    return float(value)


def _safe_std(values):
    if len(values) <= 1:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.std(unbiased=False))


def _quantile(values, q):
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.float(), q))


def matrix_geometry(matrix, name, random_eval_id=None, random_seed=None):
    matrix = matrix.float().cpu()
    col_norms = matrix.norm(p=2, dim=0)
    matrix_normed = normalize_columns(matrix)
    gram = matrix_normed.T @ matrix_normed
    eye = torch.eye(gram.size(0), dtype=torch.bool)
    off_diag = gram[~eye]
    off_abs = off_diag.abs()

    singular_values = torch.linalg.svdvals(matrix_normed)
    singular_values = singular_values.float()
    sv_sum = singular_values.sum().clamp_min(1e-12)
    sv_probs = singular_values / sv_sum
    effective_rank_entropy = torch.exp(-(sv_probs * sv_probs.clamp_min(1e-12).log()).sum())

    sv_energy = singular_values.square()
    energy_sum = sv_energy.sum().clamp_min(1e-12)
    energy_probs = sv_energy / energy_sum
    effective_rank_energy = torch.exp(
        -(energy_probs * energy_probs.clamp_min(1e-12).log()).sum()
    )
    stable_rank = energy_sum / singular_values.max().square().clamp_min(1e-12)

    row = {
        "name": name,
        "num_rows": int(matrix.size(0)),
        "num_directions": int(matrix.size(1)),
        "column_norm_mean": float(col_norms.mean()),
        "column_norm_std": float(col_norms.std(unbiased=False)),
        "column_norm_min": float(col_norms.min()),
        "column_norm_max": float(col_norms.max()),
        "cosine_offdiag_mean": float(off_diag.mean()),
        "cosine_offdiag_std": float(off_diag.std(unbiased=False)),
        "cosine_offdiag_abs_mean": float(off_abs.mean()),
        "cosine_offdiag_abs_std": float(off_abs.std(unbiased=False)),
        "cosine_offdiag_abs_p95": _quantile(off_abs, 0.95),
        "cosine_offdiag_abs_p99": _quantile(off_abs, 0.99),
        "cosine_offdiag_abs_max": float(off_abs.max()),
        "singular_value_mean": float(singular_values.mean()),
        "singular_value_std": float(singular_values.std(unbiased=False)),
        "singular_value_min": float(singular_values.min()),
        "singular_value_max": float(singular_values.max()),
        "condition_number": float(
            singular_values.max() / singular_values.min().clamp_min(1e-12)
        ),
        "effective_rank_entropy": float(effective_rank_entropy),
        "effective_rank_energy": float(effective_rank_energy),
        "stable_rank": float(stable_rank),
    }
    if random_eval_id is not None:
        row["random_eval_id"] = int(random_eval_id)
    if random_seed is not None:
        row["random_seed"] = int(random_seed)
    return row


def summarize_random_rows(rows, metric_keys):
    summary = {"num_random_eval": len(rows)}
    for key in metric_keys:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float64)
        summary[f"random_{key}_mean"] = float(tensor.mean())
        summary[f"random_{key}_std"] = float(tensor.std(unbiased=False))
        summary[f"random_{key}_min"] = float(tensor.min())
        summary[f"random_{key}_max"] = float(tensor.max())
    return summary


def compare_against_random(name, observed_row, random_rows, metric_keys, lower_is_better):
    rows = []
    for key in metric_keys:
        observed = observed_row.get(key)
        values = [row.get(key) for row in random_rows if row.get(key) is not None]
        if observed is None or not values:
            continue

        tensor = torch.tensor(values, dtype=torch.float64)
        random_mean = float(tensor.mean())
        random_std = float(tensor.std(unbiased=False))
        observed = float(observed)
        z_score = None if random_std == 0 else (observed - random_mean) / random_std
        if lower_is_better:
            better_fraction = float((tensor > observed).float().mean())
            relation = "lower_is_better"
        else:
            better_fraction = float((tensor < observed).float().mean())
            relation = "higher_is_better"

        rows.append(
            {
                "comparison": name,
                "metric": key,
                "relation": relation,
                "observed": observed,
                "random_mean": random_mean,
                "random_std": random_std,
                "random_min": float(tensor.min()),
                "random_max": float(tensor.max()),
                "observed_minus_random_mean": observed - random_mean,
                "z_score_vs_random": z_score,
                "better_than_random_fraction": better_fraction,
            }
        )
    return rows


def compare_projection_to_random(task_running_stats, random_stats):
    observed = task_running_stats["summary"]
    random_rows = [stats["summary"] for stats in random_stats]
    return compare_against_random(
        "task_running_vs_task_random",
        observed,
        random_rows,
        PROJECTION_METRICS,
        lower_is_better=True,
    )


def compare_geometry_to_random(running_geometry, random_geometry_rows):
    metric_keys = [
        key
        for key in running_geometry.keys()
        if key
        not in {
            "name",
            "num_rows",
            "num_directions",
            "random_eval_id",
            "random_seed",
        }
    ]
    lower_rows = compare_against_random(
        "running_A_geometry_vs_random_A",
        running_geometry,
        random_geometry_rows,
        [key for key in metric_keys if key in GEOMETRY_LOWER_IS_BETTER],
        lower_is_better=True,
    )
    higher_rows = compare_against_random(
        "running_A_geometry_vs_random_A",
        running_geometry,
        random_geometry_rows,
        [key for key in metric_keys if key in GEOMETRY_HIGHER_IS_BETTER],
        lower_is_better=False,
    )
    neutral_rows = compare_against_random(
        "running_A_geometry_vs_random_A",
        running_geometry,
        random_geometry_rows,
        [
            key
            for key in metric_keys
            if key not in GEOMETRY_LOWER_IS_BETTER and key not in GEOMETRY_HIGHER_IS_BETTER
        ],
        lower_is_better=True,
    )
    for row in neutral_rows:
        row["relation"] = "descriptive"
        row["better_than_random_fraction"] = None
    return lower_rows + higher_rows + neutral_rows


def main():
    args = parse_args()
    if args.num_random_eval < 1:
        raise ValueError("--num-random-eval must be at least 1.")

    device = parse_device(args.device)
    config = configure_args(load_json(args.config), device, args.seed)
    set_seed(config["seed"])
    ckpt_path = resolve_checkpoint_path(config, args)
    ckpt_obj = torch_load(ckpt_path, map_location="cpu")
    running_A, matrix_metadata = load_running_matrix(ckpt_obj, args.matrix_task_id)

    if args.inspect_matrix_only:
        inspect_matrix(ckpt_path, ckpt_obj, args.matrix_task_id)
        return

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

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
    running_A = normalize_columns(running_A)
    running_geometry = matrix_geometry(running_A, "running_A")

    all_proj, all_labels, batch_proj, batch_labels = collect_projector_outputs(
        model, loader, device, max_features=args.max_features
    )
    projector_dim = int(all_proj.size(1))
    validate_projection_setup(
        projector_dim,
        running_A,
        int(config.get("proj_output_dim", projector_dim)),
        args.num_projection_directions,
    )

    hist_directions = parse_hist_directions(args.hist_directions)
    plt = None if args.skip_figures else get_pyplot()

    batch_z_running = project_in_chunks(batch_proj, running_A, device, args.projection_chunk_size)
    batch_running_stats = compute_direction_stats(batch_z_running)
    add_sigreg(batch_running_stats, batch_z_running, args.skip_sigreg_loss, args.projection_chunk_size)
    save_stats("batch_running", batch_running_stats, output_dir)

    task_z_running = project_in_chunks(all_proj, running_A, device, args.projection_chunk_size)
    task_running_stats = compute_direction_stats(task_z_running)
    add_sigreg(task_running_stats, task_z_running, args.skip_sigreg_loss, args.projection_chunk_size)
    save_stats("task_running", task_running_stats, output_dir)

    random_stats = []
    random_geometry_rows = []
    first_random_z = None
    for eval_id in range(args.num_random_eval):
        random_seed = args.random_seed + eval_id
        A_random = generate_random_matrix(projector_dim, running_A.size(1), random_seed)
        random_geometry_rows.append(
            matrix_geometry(
                A_random,
                f"random_A_{eval_id}",
                random_eval_id=eval_id,
                random_seed=random_seed,
            )
        )
        z_random = project_in_chunks(all_proj, A_random, device, args.projection_chunk_size)
        stats = compute_direction_stats(z_random)
        add_sigreg(stats, z_random, args.skip_sigreg_loss, args.projection_chunk_size)
        stats["summary"]["random_eval_id"] = eval_id
        stats["summary"]["random_seed"] = random_seed
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
    write_rows_csv([stats["summary"] for stats in random_stats], output_dir / "task_random_multi_eval.csv")

    projection_comparison_rows = compare_projection_to_random(task_running_stats, random_stats)
    dump_json(
        {"rows": projection_comparison_rows},
        output_dir / "projection_running_vs_random_comparison.json",
    )
    write_rows_csv(
        projection_comparison_rows,
        output_dir / "projection_running_vs_random_comparison.csv",
    )

    geometry_metric_keys = [
        key
        for key in running_geometry.keys()
        if key
        not in {
            "name",
            "num_rows",
            "num_directions",
            "random_eval_id",
            "random_seed",
        }
    ]
    random_geometry_summary = summarize_random_rows(random_geometry_rows, geometry_metric_keys)
    geometry_comparison_rows = compare_geometry_to_random(running_geometry, random_geometry_rows)
    geometry_payload = {
        "running_A": running_geometry,
        "random_A_summary": random_geometry_summary,
        "random_A_per_eval": random_geometry_rows,
        "running_A_vs_random_A": geometry_comparison_rows,
    }
    dump_json(geometry_payload, output_dir / "matrix_geometry_summary.json")
    write_rows_csv([running_geometry], output_dir / "matrix_geometry_running.csv")
    write_rows_csv(random_geometry_rows, output_dir / "matrix_geometry_random_per_eval.csv")
    write_rows_csv([random_geometry_summary], output_dir / "matrix_geometry_random_summary.csv")
    write_rows_csv(geometry_comparison_rows, output_dir / "matrix_geometry_running_vs_random.csv")

    summary_rows = [
        summary_row("batch_running", batch_running_stats),
        summary_row("task_running", task_running_stats),
        random_average_row(random_stats),
    ]
    dump_json(
        {
            "metadata": {
                "checkpoint": str(ckpt_path),
                "config": args.config,
                "command": " ".join(shlex.quote(item) for item in sys.argv),
                "target_task_id": args.task_id,
                "checkpoint_task_id": checkpoint_task_id,
                "source": args.source,
                "mode": args.mode,
                "class_range": list(class_range),
                "raw_dataset_size": raw_dataset_size,
                "projector_output_dim": projector_dim,
                "running_A_shape": list(running_A.shape),
                "num_random_eval": args.num_random_eval,
                "projection_comparison": "projection_running_vs_random_comparison.csv",
                "matrix_geometry": "matrix_geometry_summary.json",
                **matrix_metadata,
            },
            "rows": summary_rows,
        },
        output_dir / "analysis_summary.json",
    )
    write_rows_csv(summary_rows, output_dir / "analysis_summary.csv")

    plot_histograms(batch_z_running, hist_directions, "batch_running", fig_dir, plt)
    plot_histograms(task_z_running, hist_directions, "task_running", fig_dir, plt)
    plot_histograms(first_random_z, hist_directions, "task_random", fig_dir, plt)
    plot_metric_per_direction(task_running_stats, "task_running", fig_dir, plt)
    plot_metric_per_direction(random_stats[0], "task_random", fig_dir, plt)
    if args.pca:
        plot_pca(all_proj, all_labels, fig_dir, args.max_pca_points, config["seed"], plt)

    print(f"Collected projector features: {list(all_proj.shape)}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Running matrix source: {matrix_metadata['matrix_source']}")
    print(f"Running matrix metadata: {matrix_metadata}")
    print(f"Batch running summary: {batch_running_stats['summary']}")
    print(f"Task running summary: {task_running_stats['summary']}")
    print(f"Random multi-eval summary: {multi_summary}")
    print(f"Projection comparison rows: {projection_comparison_rows}")
    print(f"Running matrix geometry: {running_geometry}")
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
