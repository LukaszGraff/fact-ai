from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from typing import Any, Dict

from codecarbon import EmissionsTracker


def _coerce_value(raw: str) -> Any:
    """Best-effort coercion for CLI values: bool, int, float, else str."""
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
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"Invalid --cc '{item}'. Empty key.")
        params[k] = _coerce_value(v)
    return params


def _filter_tracker_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that EmissionsTracker.__init__ actually accepts."""
    sig = inspect.signature(EmissionsTracker.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in allowed}

    unknown = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if unknown:
        print(
            f"[track_with_codecarbon] Warning: ignoring unknown EmissionsTracker params: {unknown}",
            file=sys.stderr,
        )
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Python script under CodeCarbon EmissionsTracker."
    )
    parser.add_argument(
        "--cc",
        action="append",
        default=[],
        metavar="key=value",
        help="EmissionsTracker kwarg (repeatable), e.g. --cc project_name=MyProj",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Where to store CodeCarbon CSV logs (default: ./logs/codecarbon)",
    )

    parser.add_argument("script", help="Path to the Python script to run")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the target script (passed through unchanged)",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir or os.path.join(os.getcwd(), "logs", "codecarbon")
    os.makedirs(logs_dir, exist_ok=True)

    user_kwargs = _parse_kv_list(args.cc)

    user_kwargs.setdefault("output_dir", logs_dir)
    user_kwargs.setdefault("output_file", "fairdice_runs_.csv")
    user_kwargs.setdefault("save_to_file", True)

    tracker_kwargs = _filter_tracker_kwargs(user_kwargs)

    tracker = EmissionsTracker(**tracker_kwargs)

    tracker.start()
    cmd = [sys.executable, args.script] + args.script_args

    try:
        return_code = subprocess.call(cmd)
    finally:
        tracker.stop()

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())