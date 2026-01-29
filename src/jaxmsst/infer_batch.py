import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from flax import nnx
import jax
import librosa
import numpy as np
import soundfile as sf
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxmsst.infer import collect_audio_files, load_model_from_config


@dataclass(frozen=True)
class AudioMeta:
    path: Path
    estimated_len: int
    duration_s: float


def _next_pow2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _estimate_length_samples(path: Path, target_sr: int) -> AudioMeta:
    duration_s = 0.0
    try:
        duration_s = float(librosa.get_duration(filename=str(path)))
    except Exception:
        duration_s = 0.0

    estimated_len = int(round(duration_s * target_sr)) if duration_s > 0 else 0
    if estimated_len <= 0:
        # Fallback: decode once to get a reliable length estimate.
        y, _ = librosa.load(str(path), sr=target_sr, mono=False)
        if y is None or y.shape[-1] <= 0:
            raise ValueError("解码得到空音频")
        estimated_len = int(y.shape[-1])
        duration_s = float(estimated_len) / float(target_sr)

    return AudioMeta(path=path, estimated_len=estimated_len, duration_s=duration_s)


def _ensure_channels(y: np.ndarray, expected_channels: int) -> np.ndarray:
    if y.ndim == 1:
        y = y[np.newaxis, :]

    if expected_channels == 1:
        if y.shape[0] == 1:
            return y
        return np.mean(y, axis=0, keepdims=True)

    # expected_channels == 2
    if y.shape[0] == 1:
        return np.repeat(y, 2, axis=0)
    if y.shape[0] >= 2:
        return y[:2]
    raise ValueError(f"无效的音频通道数: {y.shape}")


def _block_until_ready_pytree(pytree) -> None:
    for leaf in jax.tree_util.tree_leaves(pytree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _make_output_paths(
    audio_path: Path,
    *,
    input_root_dir: Path,
    output_root_dir: Path,
) -> Tuple[Path, str]:
    file_name = audio_path.stem
    parent_prefix = audio_path.parent.name
    output_stem = f"{parent_prefix}_{file_name}" if parent_prefix else file_name
    relative_parent = audio_path.parent.relative_to(input_root_dir) if input_root_dir in audio_path.parents else Path(".")
    file_output_dir = output_root_dir / relative_parent
    file_output_dir.mkdir(parents=True, exist_ok=True)
    return file_output_dir, output_stem


def run(args) -> None:
    start_time = time.perf_counter()

    graphdef, params, hp = load_model_from_config(
        args.config_path,
        args.start_check_point,
        model_config_path=args.model_config_path,
    )

    input_path = Path(args.input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    all_audio_paths = collect_audio_files(input_path)
    if not all_audio_paths:
        print(f"在 {input_path} 中未找到支持的音频文件")
        return

    output_root = Path(args.store_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    input_root_dir = input_path if input_path.is_dir() else input_path.parent

    target_sr = int(args.sr)
    expected_channels = 2 if bool(getattr(hp.model, "stereo", True)) else 1
    min_pad_len = int(getattr(hp.model, "stft_win_length", 2048) or 2048)

    batch_size = int(args.batch_size) if args.batch_size else int(hp.inference.batch_size)
    if batch_size <= 0:
        raise ValueError(f"无效的 batch_size: {batch_size}")
    if batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size={batch_size} 必须是 device_count={jax.device_count()} 的整数倍")

    print(f"找到音频文件总数: {len(all_audio_paths)}")
    print(f"batch_size: {batch_size}, 目标采样率: {target_sr}, 通道数: {expected_channels}")

    processed_count = 0
    failed_count = 0

    # 估算长度并按长度排序（随后按 2^n 分桶，降低 padding 和 JIT 编译次数）
    metas: List[AudioMeta] = []
    for p in all_audio_paths:
        try:
            metas.append(_estimate_length_samples(p, target_sr))
        except Exception as e:
            failed_count += 1
            print(f"  ✗ librosa.load 失败(跳过): {p}: {e}")
    metas.sort(key=lambda m: m.estimated_len)

    buckets: Dict[int, List[AudioMeta]] = {}
    for meta in metas:
        bucket_len = _next_pow2(max(meta.estimated_len, min_pad_len))
        buckets.setdefault(bucket_len, []).append(meta)

    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=("data",))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))

    with mesh:
        params = jax.device_put(params, replicate_sharding)
        _block_until_ready_pytree(params)

    @jax.jit(
        in_shardings=(replicate_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    def model_inference(params, x):
        model = nnx.merge(graphdef, params)
        return model(x, deterministic=True)

    total_batches = 0
    for bucket_len in sorted(buckets.keys()):
        total_batches += int(math.ceil(len(buckets[bucket_len]) / batch_size))
    batch_idx = 0

    for planned_pad_len in sorted(buckets.keys()):
        items = buckets[planned_pad_len]
        for offset in range(0, len(items), batch_size):
            batch_idx += 1
            batch_items = items[offset : offset + batch_size]
            display_names = [m.path.name for m in batch_items]

            batch_load_start = time.perf_counter()
            loaded_items: List[AudioMeta] = []
            mixes: List[np.ndarray] = []
            lengths: List[int] = []
            for meta in batch_items:
                try:
                    y, _ = librosa.load(str(meta.path), sr=target_sr, mono=False)
                    if y is None or y.shape[-1] <= 0:
                        raise ValueError("解码得到空音频")
                    y = _ensure_channels(y, expected_channels).astype(np.float32)
                except Exception as e:
                    failed_count += 1
                    print(f"  ✗ librosa.load 失败(跳过): {meta.path}: {e}")
                    continue
                loaded_items.append(meta)
                mixes.append(y)
                lengths.append(int(y.shape[-1]))

            actual_n = len(loaded_items)
            if actual_n == 0:
                batch_load_s = time.perf_counter() - batch_load_start
                print(
                    f"[{batch_idx}/{total_batches}] batch=0/{batch_size} "
                    f"planned_pad_len={planned_pad_len} load={batch_load_s:.2f}s (全部读取失败，跳过)"
                )
                continue

            max_len = max(lengths) if lengths else 0
            pad_len = _next_pow2(max(max_len, min_pad_len))
            if args.max_pad_len and pad_len > int(args.max_pad_len):
                raise ValueError(f"pad_len={pad_len} 超过 max_pad_len={args.max_pad_len}")

            batch_array = np.zeros((batch_size, expected_channels, pad_len), dtype=np.float32)
            for i, y in enumerate(mixes):
                batch_array[i, :, : y.shape[-1]] = y

            batch_load_s = time.perf_counter() - batch_load_start

            batch_infer_start = time.perf_counter()
            with mesh:
                out = model_inference(params, batch_array)
            if hasattr(out, "block_until_ready"):
                out.block_until_ready()
            batch_infer_s = time.perf_counter() - batch_infer_start

            out_np = np.asarray(out)
            if out_np.ndim == 3:
                out_np = out_np[:, np.newaxis, :, :]

            batch_save_start = time.perf_counter()
            for i, meta in enumerate(loaded_items):
                try:
                    file_output_dir, output_stem = _make_output_paths(
                        meta.path,
                        input_root_dir=input_root_dir,
                        output_root_dir=output_root,
                    )
                    pred = out_np[i, :, :, : lengths[i]]  # (stems, channels, time)
                    for stem_idx, instrument in enumerate(hp.training.instruments):
                        if stem_idx >= pred.shape[0]:
                            break
                        estimates = pred[stem_idx].transpose(1, 0)  # (time, channels)
                        output_file = file_output_dir / f"{output_stem}_{instrument}.wav"
                        estimates = np.nan_to_num(estimates)
                        sf.write(str(output_file), estimates, target_sr, subtype="FLOAT")
                    processed_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"  ✗ 保存失败: {meta.path.name}: {e}")

            batch_save_s = time.perf_counter() - batch_save_start

            first_loaded = loaded_items[0].path.name if loaded_items else "-"
            print(
                f"[{batch_idx}/{total_batches}] batch={actual_n}/{batch_size} "
                f"pad_len={pad_len} (planned={planned_pad_len}) "
                f"load={batch_load_s:.2f}s infer={batch_infer_s:.2f}s save={batch_save_s:.2f}s "
                f"first={first_loaded}"
            )

    elapsed_s = time.perf_counter() - start_time
    print("\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"总耗时: {elapsed_s:.2f} 秒")
    if processed_count > 0:
        print(f"平均每个文件: {elapsed_s / processed_count:.2f} 秒")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JAX音频源分离推理工具（整段 batch 推理，不分块）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.getenv("CONFIG_PATH", "./configs/infer.yaml"),
        help="推理配置文件路径",
    )
    parser.add_argument(
        "--start_check_point",
        type=str,
        default=os.getenv("START_CHECK_POINT", "deverb_bs_roformer_8_256dim_8depth.ckpt"),
        help="模型检查点文件路径",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=os.getenv("MODEL_CONFIG_PATH"),
        help="模型配置文件路径(可选)",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default=os.getenv("INPUT_FOLDER", "./input"),
        help="包含待处理音频文件的输入文件夹路径（支持递归）",
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        default=os.getenv("STORE_DIR", "./output"),
        help="分离结果保存目录路径",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "0")),
        help="batch 大小（0 表示使用配置文件 inference.batch_size）",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=int(os.getenv("TARGET_SR", "44100")),
        help="目标采样率",
    )
    parser.add_argument(
        "--max_pad_len",
        type=int,
        default=int(os.getenv("MAX_PAD_LEN", "0")),
        help="限制 padding 后的最大长度（0 表示不限制）",
    )

    args = parser.parse_args()
    if args.batch_size == 0:
        args.batch_size = None
    if args.max_pad_len == 0:
        args.max_pad_len = None

    run(args)


if __name__ == "__main__":
    main()
