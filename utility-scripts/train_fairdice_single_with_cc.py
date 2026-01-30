from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from codecarbon import EmissionsTracker


def _coerce_value(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if s.startswith(("0x", "0X")):
            return int(s, 16)
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _parse_kv_list(kv_list: list[str] | None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not kv_list:
        return params
    for item in kv_list:
        if "=" not in item:
            raise SystemExit(f"Invalid --cc '{item}'. Use key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --cc '{item}'. Empty key.")
        params[key] = _coerce_value(value)
    return params


def _filter_tracker_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(EmissionsTracker.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    unknown = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if unknown:
        print(
            "[train_fairdice_single_with_cc] Warning: ignoring unknown EmissionsTracker params: "
            f"{unknown}",
            file=sys.stderr,
        )
    return filtered


def _default_train_args() -> list[str]:
    return [
        "--learner",
        "FairDICE",
        "--divergence",
        "CHI",
        "--env_name",
        "MO-FourRoom-v2",
        "--quality",
        "amateur",
        "--beta",
        "0.001",
        "--seed",
        "1984",
        "--preference_dist",
        "uniform",
        "--eval_episodes",
        "10",
        "--batch_size",
        "128",
        "--hidden_dim",
        "256",
        "--num_layers",
        "2",
        "--total_train_steps",
        "100000",
        "--log_interval",
        "1000",
        "--normalize_reward",
        "False",
        "--max_seq_len",
        "200",
        "--policy_lr",
        "3e-4",
        "--nu_lr",
        "3e-4",
        "--mu_lr",
        "3e-4",
        "--gamma",
        "0.99",
        "--save_model_mode",
        "last",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single FairDICE training job under CodeCarbon."
    )
    parser.add_argument(
        "--cc",
        action="append",
        default=[],
        metavar="key=value",
        help="EmissionsTracker kwarg (repeatable), e.g. --cc project_name=FairDICE",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Where to store CodeCarbon CSV logs (default: ./logs/codecarbon)",
    )
    parser.add_argument(
        "--main",
        default="main_fourroom.py",
        help="Training entrypoint relative to repo root (default: main_fourroom.py)",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Skip the default FairDICE training args and rely on explicit overrides.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the training script (appended after defaults).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    main_path = Path(args.main)
    if not main_path.is_absolute():
        main_path = repo_root / main_path
    main_path = main_path.resolve()

    logs_dir = args.logs_dir or str(repo_root / "logs" / "codecarbon")
    os.makedirs(logs_dir, exist_ok=True)

    user_kwargs = _parse_kv_list(args.cc)
    user_kwargs.setdefault("project_name", "FairDICE")
    user_kwargs.setdefault("output_dir", logs_dir)
    user_kwargs.setdefault("output_file", "fairdice_runs_.csv")
    user_kwargs.setdefault("save_to_file", True)
    tracker_kwargs = _filter_tracker_kwargs(user_kwargs)

    train_args = [] if args.no_defaults else _default_train_args()
    train_args += args.train_args

    tracker = EmissionsTracker(**tracker_kwargs)
    tracker.start()
    cmd = [sys.executable, str(main_path)] + train_args

    try:
        return_code = subprocess.call(cmd)
    finally:
        tracker.stop()

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
