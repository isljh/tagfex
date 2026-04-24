import os
import shlex
import subprocess


def _run_git(args, cwd):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_git_metadata(cwd, config_path, argv, run_mode="debug"):
    metadata = {
        "run_mode": run_mode,
        "config_path": os.path.abspath(config_path) if config_path else None,
        "run_command": " ".join(shlex.quote(arg) for arg in argv),
        "git_available": False,
        "git_root": None,
        "git_branch": None,
        "git_commit": None,
        "git_dirty": None,
    }

    git_root = _run_git(["rev-parse", "--show-toplevel"], cwd)
    if git_root is None:
        return metadata

    metadata["git_available"] = True
    metadata["git_root"] = git_root
    metadata["git_branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    metadata["git_commit"] = _run_git(["rev-parse", "HEAD"], cwd)

    status = _run_git(["status", "--porcelain"], cwd)
    metadata["git_dirty"] = bool(status) if status is not None else None
    return metadata


def validate_git_run_mode(metadata):
    run_mode = metadata.get("run_mode", "debug")
    git_available = metadata.get("git_available", False)
    git_dirty = metadata.get("git_dirty", None)

    if run_mode not in {"debug", "exp"}:
        raise ValueError("run_mode must be either 'debug' or 'exp'")

    if run_mode == "exp":
        if not git_available:
            raise RuntimeError(
                "Formal experiment mode requires a Git repository, but no Git repo was detected."
            )
        if git_dirty:
            raise RuntimeError(
                "Formal experiment mode requires a clean Git workspace. "
                "Please commit or stash your changes before running."
            )
