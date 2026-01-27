from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

_TRAINING_KEYS = ("train", "data", "data_loader", "log", "inference")


def _resolve_relative(base_path: str, maybe_path: str) -> Path:
    candidate = Path(maybe_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return Path(base_path).parent / candidate


def _infer_model_config_path(
    base_config: DictConfig,
    config_path: str,
    model_config_path: Optional[str],
    checkpoint_path: Optional[str],
) -> Optional[Path]:
    if model_config_path:
        return _resolve_relative(config_path, model_config_path)

    if "model_config_path" in base_config and base_config.model_config_path:
        return _resolve_relative(config_path, str(base_config.model_config_path))

    if "model" in base_config:
        model_section = base_config.model
        if isinstance(model_section, DictConfig) and "config_path" in model_section:
            candidate = model_section.config_path
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
) -> DictConfig:
    base_path = Path(config_path)
    if not base_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    base_config = OmegaConf.load(config_path)
    model_path = _infer_model_config_path(
        base_config, config_path, model_config_path, checkpoint_path
    )

    if not model_path:
        return base_config

    if not model_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {model_path}")

    model_config = OmegaConf.load(str(model_path))

    # Merge with model config and keep training/inference from the configs folder.
    merged = OmegaConf.merge(base_config, model_config)
    for key in _TRAINING_KEYS:
        if key in base_config:
            merged[key] = base_config[key]
    return merged
