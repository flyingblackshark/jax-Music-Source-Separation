import argparse
import librosa
import numpy as np
import jax.numpy as jnp
import soundfile as sf
import glob
import os
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from functools import partial
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
import time
from omegaconf import OmegaConf
cc.set_cache_dir("/tmp/jit_cache")
def load_model_from_config(config_path,start_check_point):
    hp = OmegaConf.load(config_path)
    model = None
    params = None
    match hp.model.type:
        case "bs_roformer":
            from jaxmsst.models.bs_roformer import BSRoformer
            from convert import load_bs_roformer_params
            model = BSRoformer(dim=hp.model.dim,
                                depth=hp.model.depth,
                                stereo=hp.model.stereo,
                                num_stems=hp.model.num_stems,
                                use_shared_bias=hp.model.use_shared_bias,
                                time_transformer_depth=hp.model.time_transformer_depth,
                                freq_transformer_depth=hp.model.freq_transformer_depth)
            params = load_bs_roformer_params(start_check_point,hp)
        case "mel_band_roformer":
            from jaxmsst.models.mel_band_roformer import MelBandRoformer
            from convert import load_mel_band_roformer_params
            model = MelBandRoformer(dim=hp.model.dim,
                                    depth=hp.model.depth,
                                    stereo=hp.model.stereo,
                                    time_transformer_depth=hp.model.time_transformer_depth,
                                    freq_transformer_depth=hp.model.freq_transformer_depth)
            params = load_mel_band_roformer_params(start_check_point,hp)
        case _:
            raise Exception("unknown model")
    return model,params,hp
def run_folder(args):
    start_time = time.time()
    model,params,hp = load_model_from_config(args.config_path,args.start_check_point)
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    all_mixtures_path.sort()
    print('Total files found: {}'.format(len(all_mixtures_path)))

    # instruments = config.training.instruments
    # if config.training.target_instrument is not None:
    #     instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    # if not verbose:
    #     all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    for path in all_mixtures_path:
        print("Starting processing track: ", path)
        try:
            mix, sr = librosa.load(path, sr=44100, mono=False)
        except Exception as e:
            print('Can read track: {}'.format(path))
            print('Error message: {}'.format(str(e)))
            continue

        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)

        #mix_orig = mix.copy()

        res = demix_track(model,params,mix,mesh,hp)

        file_name, _ = os.path.splitext(os.path.basename(path))
        
        for i in range(len(hp.model.instruments)):
            estimates = res[i].transpose(1,0)
            output_file = os.path.join(args.store_dir, f"{file_name}_{hp.model.instruments[i]}.wav")
            sf.write(output_file, estimates, sr, subtype = 'FLOAT')

        # instrum_file_name = os.path.join(args.store_dir, f"{file_name}_other.wav")
        # sf.write(instrum_file_name, mix_orig.T - res.sum(0).transpose(1,0), sr, subtype = 'FLOAT')

    #time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def demix_track(model, params, mix, mesh, hp):
    """优化的音频分离函数
    
    Args:
        model: JAX模型
        params: 模型参数
        mix: 输入音频混合信号 (channels, samples)
        mesh: JAX设备网格
        hp: 超参数配置
    
    Returns:
        estimated_sources: 分离后的音频源 (num_stems, channels, samples)
    """
    # 提取配置参数
    C = hp.inference.chunk_size
    N = hp.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = hp.inference.batch_size
    
    # 输入验证
    if mix.ndim != 2:
        raise ValueError(f"Expected 2D input (channels, samples), got {mix.ndim}D")
    
    length_init = mix.shape[-1]
    
    # 设置分片策略
    x_sharding = NamedSharding(mesh, PartitionSpec('data'))
    
    # JIT编译的模型推理函数
    @partial(jax.jit, in_shardings=(None, x_sharding), out_shardings=x_sharding)
    def model_apply(params, x):
        return model.apply({'params': params}, x, deterministic=True)
    
    # 优化的窗口化函数
    @partial(jax.jit, in_shardings=(x_sharding, None), out_shardings=x_sharding)
    def apply_windowing_vmap(x_batch, window):
        return jax.vmap(lambda x: x * window)(x_batch)
    
    # 预计算窗口数组
    def _create_windowing_array(window_size: int, fade_size: int) -> np.ndarray:
        """创建带淡入淡出的窗口数组"""
        window = np.ones(window_size, dtype=np.float32)
        if fade_size > 0:
            fadein = np.linspace(0, 1, fade_size, dtype=np.float32)
            fadeout = np.linspace(1, 0, fade_size, dtype=np.float32)
            window[:fade_size] *= fadein
            window[-fade_size:] *= fadeout
        return window
    
    # 预计算不同类型的窗口
    base_window = _create_windowing_array(C, fade_size)
    first_window = base_window.copy()
    first_window[:fade_size] = 1.0  # 第一个块无淡入
    last_window = base_window.copy()
    last_window[-fade_size:] = 1.0  # 最后一个块无淡出
    
    # 音频预处理：边界填充
    if length_init > 2 * border and border > 0:
        mix = np.pad(mix, ((0, 0), (border, border)), mode='reflect')
    
    # 初始化结果数组
    req_shape = (hp.model.num_stems,) + tuple(mix.shape)
    result = np.zeros(req_shape, dtype=np.float32)
    counter = np.zeros(req_shape, dtype=np.float32)
    
    # 批处理变量
    batch_data = []
    batch_locations = []
    i = 0
    total_chunks = (mix.shape[1] - C) // step + 1
    
    # 主处理循环
    while i < mix.shape[1]:
        # 提取音频块
        part = mix[:, i:i + C]
        length = part.shape[-1]
        
        # 处理不完整的块
        if length < C:
            pad_mode = 'reflect' if length > C // 2 + 1 else 'constant'
            part = np.pad(part, ((0, 0), (0, C - length)), mode=pad_mode)
        
        batch_data.append(part)
        batch_locations.append((i, length))
        i += step
        
        # 批处理推理
        if len(batch_data) >= batch_size or i >= mix.shape[1]:
            current_batch_size = len(batch_data)
            
            # 准备批次数据
            arr = np.stack(batch_data, axis=0)
            if current_batch_size < batch_size:
                # 填充到批次大小
                padding_size = batch_size - current_batch_size
                arr = np.pad(arr, ((0, padding_size), (0, 0), (0, 0)))
            
            # 模型推理
            with mesh:
                arr_jax = jnp.asarray(arr)
                x = model_apply(params, arr_jax)
            
            # 选择合适的窗口
            chunk_idx = (i - step) // step
            if chunk_idx == 0:
                window = first_window
            elif i >= mix.shape[1]:
                window = last_window
            else:
                window = base_window
            
            # 应用窗口化
            windowed_output = apply_windowing_vmap(x[..., :C], window)
            windowed_output = windowed_output[:current_batch_size]
            windowed_output = np.asarray(windowed_output)
            
            # 累加结果
            for j, (start, length) in enumerate(batch_locations):
                end = start + length
                result[..., start:end] += windowed_output[j][..., :length]
                counter[..., start:end] += window[:length]
            
            # 清空批次
            batch_data.clear()
            batch_locations.clear()
    
    # 后处理：归一化和去除填充
    with np.errstate(divide='ignore', invalid='ignore'):
        estimated_sources = np.divide(result, counter, 
                                    out=np.zeros_like(result), 
                                    where=counter != 0)
    
    # 移除边界填充
    if length_init > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    
    return estimated_sources

def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.getenv('CONFIG_PATH', './configs/bs_roformer_base.yaml'),
                        help="path to config file")
    parser.add_argument("--start_check_point", type=str,
                        default=os.getenv('START_CHECK_POINT', 'deverb_bs_roformer_8_256dim_8depth.ckpt'),
                        help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, default=os.getenv('INPUT_FOLDER', './input'),
                        help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, default=os.getenv('STORE_DIR', './output'),
                        help="path to store results as wav file")
    args = parser.parse_args()
    run_folder(args)


if __name__ == "__main__":
    main()
