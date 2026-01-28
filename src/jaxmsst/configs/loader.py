from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

_TRAINING_KEYS = ("train", "data", "data_loader", "log", "inference")

class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _to_attrdict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attrdict(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    result = {k: v for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    data = yaml.load(path.read_text(), Loader=yaml.FullLoader)
    return data or {}


def _infer_model_type(model_config: dict) -> str:
    if "model" in model_config and "num_bands" in model_config["model"]:
        return "mel_band_roformer"
    return "bs_roformer"


def _resolve_relative(base_path: str, maybe_path: str) -> Path:
    candidate = Path(maybe_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return Path(base_path).parent / candidate


def _infer_model_config_path(
    base_config: dict,
    config_path: str,
    model_config_path: Optional[str],
    checkpoint_path: Optional[str],
) -> Optional[Path]:
    if model_config_path:
        return _resolve_relative(config_path, model_config_path)

    if "model_config_path" in base_config and base_config["model_config_path"]:
        return _resolve_relative(config_path, str(base_config["model_config_path"]))

    if "model" in base_config:
        model_section = base_config.get("model") or {}
        if isinstance(model_section, dict) and "config_path" in model_section:
            candidate = model_section.get("config_path")
            if candidate:
                return _resolve_relative(config_path, str(candidate))

    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if checkpoint.is_file():
            candidate = checkpoint.parent / "config.yaml"
        else:
            candidate = checkpoint / "config.yaml"
        if candidate.exists():
            return candidate

    return None


def load_config(
    config_path: str,
    model_config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> AttrDict:
    base_path = Path(config_path)
    if not base_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    base_config = _load_yaml(base_path)
    model_path = _infer_model_config_path(
        base_config, config_path, model_config_path, checkpoint_path
    )

    if not model_path:
        return _to_attrdict(base_config)

    if not model_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {model_path}")

    model_config = _load_yaml(model_path)

    # Merge with model config and keep training/inference from the configs folder.
    merged = _deep_merge(model_config, base_config)
    for key in _TRAINING_KEYS:
        if key in base_config:
            merged[key] = base_config[key]
    return _to_attrdict(merged)


def load_infer_config(
    infer_config_path: str,
    model_config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> AttrDict:
    infer_path = Path(infer_config_path)
    if not infer_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {infer_config_path}")

    infer_config = _load_yaml(infer_path)
    model_path = _infer_model_config_path(
        infer_config, infer_config_path, model_config_path, checkpoint_path
    )
    if not model_path:
        raise ValueError("未提供模型配置文件路径或无法从检查点推断模型配置路径")
    if not model_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {model_path}")

    model_config = _load_yaml(model_path)
    merged = dict(model_config)

    if "inference" in infer_config:
        merged["inference"] = infer_config["inference"]

    if "model" not in merged:
        merged["model"] = {}

    if "type" not in merged["model"]:
        if "model" in infer_config and "type" in infer_config["model"]:
            merged["model"]["type"] = infer_config["model"]["type"]
        else:
            merged["model"]["type"] = _infer_model_type(model_config)

    return _to_attrdict(merged)
