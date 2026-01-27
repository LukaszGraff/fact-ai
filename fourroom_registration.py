import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Type

try:
    import gymnasium as gymnasium
except ModuleNotFoundError:
    gymnasium = None

try:
    import gym
except ModuleNotFoundError:
    gym = None

_FOURROOM_ENV_ID = "MO-FourRoom-v2"


def _module_path() -> Path:
    return Path(__file__).with_name("MO-Four-Room.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "claim2_experiment_fig7_fourroom", os.fspath(_module_path())
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError("Unable to load MO-Four-Room environment module")


@lru_cache(maxsize=1)
def _fourroom_class() -> Type:
    module = _load_module()
    if not hasattr(module, "FourRoomEnv"):
        raise AttributeError("FourRoomEnv not found in MO-Four-Room.py")
    return module.FourRoomEnv


def _remove_stale_registration(registry):
    if isinstance(registry, dict):
        registry.pop(_FOURROOM_ENV_ID, None)
        return
    for attr in ("env_specs", "specs"):
        env_specs = getattr(registry, attr, None)
        if env_specs is not None and _FOURROOM_ENV_ID in env_specs:
            env_specs.pop(_FOURROOM_ENV_ID, None)


def _register_with(gym_module):
    registry = gym_module.envs.registration.registry
    _remove_stale_registration(registry)
    gym_module.envs.registration.register(
        id=_FOURROOM_ENV_ID,
        entry_point=_fourroom_class(),
        max_episode_steps=500,
    )


def ensure_fourroom_registered():
    """Register MO-FourRoom-v2 in both gymnasium and gym (if available)."""
    if gymnasium is not None:
        _register_with(gymnasium)
    if gym is not None:
        _register_with(gym)
