import argparse
import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple

from flax import nnx
import jax
import librosa
import numpy as np
import soundfile as sf
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from jaxmsst.configs.loader import load_infer_config

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac"}

@dataclass
class TimingStats:
    times_s: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times_s[name] += time.perf_counter() - start
            self.counts[name] += 1

    def merge(self, other: "TimingStats") -> None:
        for k, v in other.times_s.items():
            self.times_s[k] += v
        for k, v in other.counts.items():
            self.counts[k] += v

    def format_lines(
        self,
        *,
        min_s: float = 0.0,
        keys: Optional[Iterable[str]] = None,
        indent: str = "  ",
    ) -> List[str]:
        if keys is None:
            items = [(k, v) for k, v in self.times_s.items() if v >= min_s]
        else:
            items = [(k, self.times_s.get(k, 0.0)) for k in keys if self.times_s.get(k, 0.0) >= min_s]

        items.sort(key=lambda kv: kv[0])
        lines: List[str] = []
        for name, seconds in items:
            count = self.counts.get(name, 0)
            if count > 1 and seconds > 0:
                lines.append(f"{indent}{name}: {seconds:.3f}s ({count}x, {seconds / count:.3f}s/次)")
            else:
                lines.append(f"{indent}{name}: {seconds:.3f}s")
        return lines


def _block_until_ready_pytree(pytree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(pytree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _is_device_pytree(pytree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(pytree)
    if not leaves:
        return False
    return all(hasattr(leaf, "block_until_ready") for leaf in leaves)


def load_model_from_config(
    config_path: str,
    start_check_point: str,
    model_config_path: Optional[str] = None,
) -> Tuple[nnx.GraphDef, dict, dict]:
    """加载模型配置和参数
    
    Args:
        config_path: 配置文件路径
        start_check_point: 检查点文件路径
        model_config_path: 模型配置文件路径(可选)
        
    Returns:
        Tuple[graphdef, params, hp]: 模型定义、参数和超参数配置
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 未知的模型类型
    """
    hp = load_infer_config(
        config_path,
        model_config_path=model_config_path,
        checkpoint_path=start_check_point,
    )
    
    if not hasattr(hp, 'model') or not hasattr(hp.model, 'type'):
        raise ValueError("配置文件中缺少模型类型定义")
    
    model_type = hp.model.type.lower()
    attention_type = getattr(hp.model, "attention", "dot_product")
    flash_min_seq_len = int(getattr(hp.model, "flash_min_seq_len", 0) or 0)
    
    if model_type == "bs_roformer":
        from jaxmsst.models.bs_roformer import BSRoformer
        from jaxmsst.convert import load_bs_roformer_params
        
        model = BSRoformer(
            dim=hp.model.dim,
            depth=hp.model.depth,
            stereo=hp.model.stereo,
            num_stems=hp.model.num_stems,
            time_transformer_depth=hp.model.time_transformer_depth,
            freq_transformer_depth=hp.model.freq_transformer_depth,
            attention_type=attention_type,
            flash_min_seq_len=flash_min_seq_len,
            rngs=nnx.Rngs(0),
        )
        params = load_bs_roformer_params(start_check_point, hp)
        
    elif model_type == "mel_band_roformer":
        from jaxmsst.models.mel_band_roformer import MelBandRoformer
        from jaxmsst.convert import load_mel_band_roformer_params
        
        model = MelBandRoformer(
            dim=hp.model.dim,
            depth=hp.model.depth,
            stereo=hp.model.stereo,
            time_transformer_depth=hp.model.time_transformer_depth,
            freq_transformer_depth=hp.model.freq_transformer_depth,
            attention_type=attention_type,
            flash_min_seq_len=flash_min_seq_len,
            rngs=nnx.Rngs(0),
        )
        params = load_mel_band_roformer_params(start_check_point, hp)
        
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
        
    graphdef, _ = nnx.split(model, nnx.Param)
    return graphdef, params, hp


def collect_audio_files(input_path: Path) -> List[Path]:
    """收集输入路径下的音频文件（支持递归遍历子目录）"""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES else []

    if not input_path.is_dir():
        return []

    audio_files = [
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    return sorted(audio_files, key=lambda p: str(p))


def run_folder(args) -> None:
    """批量处理文件夹中的音频文件
    
    Args:
        args: 命令行参数对象
    """
    run_stats = TimingStats()
    start_time = time.perf_counter()
    
    try:
        with run_stats.time("setup/load_model"):
            graphdef, params, hp = load_model_from_config(
                args.config_path,
                args.start_check_point,
                model_config_path=args.model_config_path,
            )
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 获取输入文件列表
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"输入文件夹不存在: {args.input_folder}")
        return
        
    # 支持递归遍历子目录
    all_mixtures_path = collect_audio_files(input_path)
    
    if not all_mixtures_path:
        print(f"在 {args.input_folder} 中未找到支持的音频文件")
        return
        
    print(f'找到音频文件总数: {len(all_mixtures_path)}')

    # 创建输出目录
    output_path = Path(args.store_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_root_dir = input_path if input_path.is_dir() else input_path.parent

    # 初始化JAX设备网格
    with run_stats.time("setup/init_mesh"):
        device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
        mesh = Mesh(devices=device_mesh, axis_names=('data',))

    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec('data'))

    # 提前放置参数到设备（避免每个文件重复传输）
    with run_stats.time("setup/device_put_params"):
        with mesh:
            params = jax.device_put(params, replicate_sharding)
            _block_until_ready_pytree(params)

    # 只创建一次 JIT 推理函数（避免每个文件重复编译）
    @partial(
        jax.jit,
        in_shardings=(replicate_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    def model_inference(params, x):
        model = nnx.merge(graphdef, params)
        return model(x, deterministic=True)

    compile_state: MutableMapping[str, bool] = {"first_inference": True}
    
    # 处理统计
    processed_count = 0
    failed_count = 0
    accumulated_stats = TimingStats()
    
    for idx, audio_path in enumerate(all_mixtures_path, 1):
        file_stats = TimingStats()
        try:
            display_path = str(audio_path.relative_to(input_root_dir))
        except Exception:
            display_path = audio_path.name
        print(f"[{idx}/{len(all_mixtures_path)}] 处理音频: {display_path}")
        
        try:
            with file_stats.time("file/total"):
                # 加载音频文件
                with file_stats.time("file/load_audio"):
                    mix, sr = librosa.load(str(audio_path), sr=44100, mono=False)

                # 确保立体声格式
                with file_stats.time("file/preprocess_audio"):
                    if mix.ndim == 1:
                        mix = np.stack([mix, mix], axis=0)
                    elif mix.ndim == 2 and mix.shape[0] > 2:
                        # 如果有多个声道，只取前两个
                        mix = mix[:2]

                # 执行音频分离
                with file_stats.time("file/demix_total"):
                    separated_sources = demix_track(
                        graphdef,
                        params,
                        mix,
                        mesh,
                        hp,
                        model_inference=model_inference,
                        timings=file_stats,
                        compile_state=compile_state,
                    )

                # 保存分离结果
                with file_stats.time("file/save_outputs"):
                    file_name = audio_path.stem
                    parent_prefix = audio_path.parent.name
                    output_stem = f"{parent_prefix}_{file_name}" if parent_prefix else file_name
                    relative_parent = (
                        audio_path.parent.relative_to(input_root_dir) if input_path.is_dir() else Path(".")
                    )
                    file_output_dir = output_path / relative_parent
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, instrument in enumerate(hp.training.instruments):
                        if i < len(separated_sources):
                            estimates = separated_sources[i].transpose(1, 0)
                            output_file = file_output_dir / f"{output_stem}_{instrument}.wav"
                            sf.write(str(output_file), estimates, sr, subtype='FLOAT')
            
            processed_count += 1
            accumulated_stats.merge(file_stats)
            total_s = file_stats.times_s.get("file/total", 0.0)
            load_s = file_stats.times_s.get("file/load_audio", 0.0)
            pre_s = file_stats.times_s.get("file/preprocess_audio", 0.0)
            demix_s = file_stats.times_s.get("file/demix_total", 0.0)
            save_s = file_stats.times_s.get("file/save_outputs", 0.0)
            print(
                "  ✓ 处理完成"
                f" | 总计 {total_s:.2f}s"
                f" (读取 {load_s:.2f}s, 预处理 {pre_s:.2f}s, 推理 {demix_s:.2f}s, 保存 {save_s:.2f}s)"
            )
            if args.verbose:
                demix_lines = file_stats.format_lines(
                    min_s=0.001,
                    keys=[
                        "demix/allocate_buffers",
                        "demix/pad_mix",
                        "demix/batch_prepare",
                        "demix/model_inference_compile+run",
                        "demix/model_inference_run",
                        "demix/windowing_and_accumulate",
                        "demix/postprocess_normalize",
                        "demix/remove_padding",
                    ],
                    indent="    ",
                )
                if demix_lines:
                    print("    demix 细分耗时:")
                    print("\n".join(demix_lines))
            
        except Exception as e:
            failed_count += 1
            print(f"  ✗ 处理失败: {e}")
            continue

    # 输出处理统计
    elapsed_time = time.perf_counter() - start_time
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    if processed_count > 0:
        print(f"平均每个文件: {elapsed_time/processed_count:.2f} 秒")
    print("\n耗时统计(准备阶段):")
    print("\n".join(run_stats.format_lines(min_s=0.001, indent="  ")))
    if processed_count > 0:
        print("\n耗时统计(文件累计):")
        print("\n".join(accumulated_stats.format_lines(min_s=0.001, indent="  ")))


def demix_track(
    graphdef,
    params,
    mix: np.ndarray,
    mesh: Mesh,
    hp: dict,
    *,
    model_inference: Optional[Callable[[Any, Any], Any]] = None,
    timings: Optional[TimingStats] = None,
    compile_state: Optional[MutableMapping[str, bool]] = None,
) -> np.ndarray:
    """音频分离核心函数
    
    使用滑动窗口和批处理技术对音频进行源分离，支持GPU加速和内存优化。
    
    Args:
        model: JAX模型实例
        params: 模型参数字典
        mix: 输入音频混合信号，形状为 (channels, samples)
        mesh: JAX设备网格，用于分布式计算
        hp: 超参数配置对象
    
    Returns:
        estimated_sources: 分离后的音频源，形状为 (num_stems, channels, samples)
        
    Raises:
        ValueError: 输入音频维度不正确
    """
    timeit = timings.time if timings is not None else (lambda _name: nullcontext())
    # 提取和验证配置参数
    chunk_size = hp.inference.chunk_size
    num_overlap = hp.inference.num_overlap
    fade_size = chunk_size // 10
    step_size = chunk_size // num_overlap
    border_size = chunk_size - step_size
    batch_size = hp.inference.batch_size
    
    # 输入验证
    if mix.ndim != 2:
        raise ValueError(f"输入音频应为2维 (channels, samples)，实际为 {mix.ndim} 维")
    
    if mix.shape[0] > 2:
        print(f"警告: 输入音频有 {mix.shape[0]} 个声道，将只使用前2个声道")
        mix = mix[:2]
    
    original_length = mix.shape[-1]
    
    # 验证配置参数的合理性
    if chunk_size <= 0 or step_size <= 0:
        raise ValueError(f"无效的块大小配置: chunk_size={chunk_size}, step_size={step_size}")
    
    if batch_size <= 0:
        raise ValueError(f"无效的批处理大小: {batch_size}")
    
    # 设置JAX分片策略
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec('data'))
    
    # 将参数放置到设备上（上游已处理则跳过）
    if not _is_device_pytree(params):
        with timeit("demix/device_put_params"):
            with mesh:
                params = jax.device_put(params, replicate_sharding)
                _block_until_ready_pytree(params)

    if model_inference is None:
        # JIT编译的模型推理函数（批量处理建议在外部创建并复用）
        @partial(
            jax.jit,
            in_shardings=(replicate_sharding, data_sharding),
            out_shardings=data_sharding,
        )
        def model_inference(params, x):
            model = nnx.merge(graphdef, params)
            return model(x, deterministic=True)
    
    # 窗口化函数改为主机端处理，避免小批次与多设备分片冲突
    def apply_windowing_single(x_single, window):
        """单个块应用窗口函数"""
        return np.asarray(x_single) * window
    
    def _create_fade_window(window_size: int, fade_size: int) -> np.ndarray:
        """创建带淡入淡出效果的窗口数组
        
        Args:
            window_size: 窗口大小
            fade_size: 淡入淡出大小
            
        Returns:
            窗口数组
        """
        window = np.ones(window_size, dtype=np.float32)
        if fade_size > 0 and fade_size < window_size // 2:
            # 淡入效果
            fade_in = np.linspace(0, 1, fade_size, dtype=np.float32)
            window[:fade_size] = fade_in
            # 淡出效果
            fade_out = np.linspace(1, 0, fade_size, dtype=np.float32)
            window[-fade_size:] = fade_out
        return window
    
    # 预计算窗口数组
    base_window = _create_fade_window(chunk_size, fade_size)
    first_window = base_window.copy()
    first_window[:fade_size] = 1.0  # 第一个块不需要淡入
    last_window = base_window.copy()
    last_window[-fade_size:] = 1.0  # 最后一个块不需要淡出
    
    # 音频预处理：边界填充以减少边界效应
    with timeit("demix/pad_mix"):
        if original_length > 2 * border_size and border_size > 0:
            mix = np.pad(mix, ((0, 0), (border_size, border_size)), mode='reflect')
    
    # 初始化结果累积数组
    with timeit("demix/allocate_buffers"):
        result_shape = (hp.model.num_stems,) + tuple(mix.shape)
        accumulated_result = np.zeros(result_shape, dtype=np.float32)
        overlap_counter = np.zeros(result_shape, dtype=np.float32)
    
    # 批处理状态变量
    batch_chunks = []
    batch_positions = []
    current_position = 0
    
    # 计算总块数用于进度显示
    total_chunks = max(1, (mix.shape[1] - chunk_size) // step_size + 1)
    
    # 主处理循环：滑动窗口处理
    while current_position < mix.shape[1]:
        # 提取当前音频块
        chunk_end = min(current_position + chunk_size, mix.shape[1])
        audio_chunk = mix[:, current_position:chunk_end]
        actual_length = audio_chunk.shape[-1]
        
        # 处理不完整的块（末尾块）
        if actual_length < chunk_size:
            # 根据块的大小选择填充策略
            pad_mode = 'reflect' if actual_length > chunk_size // 2 else 'constant'
            pad_width = ((0, 0), (0, chunk_size - actual_length))
            audio_chunk = np.pad(audio_chunk, pad_width, mode=pad_mode)
        
        # 添加到批处理队列
        batch_chunks.append(audio_chunk)
        batch_positions.append((current_position, actual_length))
        current_position += step_size
        
        # 执行批处理推理
        if len(batch_chunks) >= batch_size or current_position >= mix.shape[1]:
            current_batch_size = len(batch_chunks)
            
            # 准备批处理数据
            with timeit("demix/batch_prepare"):
                batch_array = np.stack(batch_chunks, axis=0)
                
                # 如果批次不满，填充到完整批次大小
                if current_batch_size < batch_size:
                    padding_needed = batch_size - current_batch_size
                    pad_shape = ((0, padding_needed), (0, 0), (0, 0))
                    batch_array = np.pad(batch_array, pad_shape, mode='constant')
            
            # 执行模型推理
            try:
                with mesh:
                    is_first = bool(compile_state.get("first_inference")) if compile_state else False
                    key = "demix/model_inference_compile+run" if is_first else "demix/model_inference_run"
                    with timeit(key):
                        inference_output = model_inference(params, batch_array)
                        if hasattr(inference_output, "block_until_ready"):
                            inference_output.block_until_ready()
                    if compile_state is not None and is_first:
                        compile_state["first_inference"] = False
                
                # 确定当前批次使用的窗口类型
                batch_start_idx = (current_position - step_size * current_batch_size) // step_size
                
                # 为每个块选择合适的窗口
                with timeit("demix/windowing_and_accumulate"):
                    for j in range(current_batch_size):
                        chunk_idx = batch_start_idx + j
                        
                        if chunk_idx == 0:
                            window = first_window
                        elif current_position >= mix.shape[1] and j == current_batch_size - 1:
                            window = last_window
                        else:
                            window = base_window
                        
                        # 应用窗口化到单个输出（转回主机端数组）
                        single_output = inference_output[j:j+1, ..., :chunk_size]
                        windowed_result = apply_windowing_single(single_output[0], window)
                        
                        # 累加到结果数组
                        start_pos, actual_len = batch_positions[j]
                        end_pos = start_pos + actual_len
                        
                        accumulated_result[..., start_pos:end_pos] += windowed_result[..., :actual_len]
                        overlap_counter[..., start_pos:end_pos] += window[:actual_len]
                
            except Exception as e:
                print(f"批处理推理失败: {e}")
                raise
            
            # 清空批处理队列
            batch_chunks.clear()
            batch_positions.clear()
    
    # 后处理：归一化重叠区域
    with timeit("demix/postprocess_normalize"):
        with np.errstate(divide='ignore', invalid='ignore'):
            # 避免除零错误，使用安全除法
            estimated_sources = np.divide(
                accumulated_result, 
                overlap_counter,
                out=np.zeros_like(accumulated_result), 
                where=overlap_counter != 0
            )
    
    # 移除边界填充，恢复原始长度
    with timeit("demix/remove_padding"):
        if original_length > 2 * border_size and border_size > 0:
            estimated_sources = estimated_sources[..., border_size:-border_size]
    
    # 确保输出长度与原始输入一致
    if estimated_sources.shape[-1] != original_length:
        estimated_sources = estimated_sources[..., :original_length]
    
    return estimated_sources

def main() -> None:
    """音频源分离推理脚本的主入口点
    
    支持批量处理音频文件，使用JAX进行GPU加速推理。
    支持的音频格式：WAV, MP3, FLAC, M4A, AAC
    """
    parser = argparse.ArgumentParser(
        description='JAX音频源分离推理工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config_path', 
        type=str, 
        default=os.getenv('CONFIG_PATH', './configs/infer.yaml'),
        help='推理配置文件路径'
    )
    
    parser.add_argument(
        '--start_check_point', 
        type=str,
        default=os.getenv('START_CHECK_POINT', 'deverb_bs_roformer_8_256dim_8depth.ckpt'),
        help='模型检查点文件路径'
    )

    parser.add_argument(
        '--model_config_path',
        type=str,
        default=os.getenv('MODEL_CONFIG_PATH'),
        help='模型配置文件路径(可选)',
    )
    
    parser.add_argument(
        '--input_folder', 
        type=str, 
        default=os.getenv('INPUT_FOLDER', './input'),
        help='包含待处理音频文件的输入文件夹路径'
    )
    
    parser.add_argument(
        '--store_dir', 
        type=str, 
        default=os.getenv('STORE_DIR', './output'),
        help='分离结果保存目录路径'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='显示详细处理信息'
    )
    
    args = parser.parse_args()
    
    # 参数验证
    if not Path(args.config_path).exists():
        print(f"错误: 配置文件不存在: {args.config_path}")
        return
    
    if not Path(args.input_folder).exists():
        print(f"错误: 输入文件夹不存在: {args.input_folder}")
        return
    
    # 显示运行信息
    if args.verbose:
        print(f"配置文件: {args.config_path}")
        print(f"检查点文件: {args.start_check_point}")
        print(f"输入文件夹: {args.input_folder}")
        print(f"输出文件夹: {args.store_dir}")
        print(f"可用JAX设备数: {jax.device_count()}")
        print()
    
    try:
        run_folder(args)
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
