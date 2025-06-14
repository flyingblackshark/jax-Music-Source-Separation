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
cc.set_cache_dir("./jax_cache")
def load_model_from_config(config_path,start_check_point):
    hp = OmegaConf.load(config_path)
    model = None
    params = None
    match hp.model.type:
        case "bs_roformer":
            from models.bs_roformer import BSRoformer
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
            from models.mel_band_roformer import MelBandRoformer
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


def demix_track(model,params, mix,mesh, hp):
    #default chunk size 
    C = hp.inference.chunk_size
    N = hp.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = hp.inference.batch_size

    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    @partial(jax.jit, in_shardings=(None,x_sharding),
                    out_shardings=x_sharding)
    def model_apply(params, x):
        return model.apply({'params': params}, x , deterministic=True)
    
    # 使用vmap优化的窗口化函数
    @partial(jax.jit, in_shardings=(x_sharding, None), out_shardings=x_sharding)
    def apply_windowing_vmap(x_batch, window):
        return jax.vmap(lambda x: x * window)(x_batch)
    
    # 使用vmap优化的结果累加函数
    @jax.jit
    def accumulate_results_vmap(results, windows, starts, lengths):
        def accumulate_single(result, window, start, length):
            return result[:length], window[:length], start
        return jax.vmap(accumulate_single)(results, windows, starts, lengths)
    
    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix = np.pad(mix, ((0,0),(border, border)), mode='reflect')
    
    def _getWindowingArray(window_size, fade_size):
        fadein = np.linspace(0, 1, fade_size)
        fadeout = np.linspace(1, 0, fade_size)
        window = np.ones(window_size)
        window[-fade_size:] = (window[-fade_size:]*fadeout)
        window[:fade_size] = (window[:fade_size]*fadein)
        return window
    
    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

    req_shape = (hp.model.num_stems, ) + tuple(mix.shape)
    result = np.zeros(req_shape, dtype=jnp.float32)
    counter = np.zeros(req_shape, dtype=jnp.float32)
    i = 0
    batch_data = []
    batch_locations = []

    while i < mix.shape[1]:
        part = mix[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = np.pad(part,((0,0),(0,C-length)),mode='reflect')
            else:
                part = np.pad(part,((0,0),(0,C-length)),mode='constant')
        batch_data.append(part)
        batch_locations.append((i, length))
        i += step

        if len(batch_data) >= batch_size or (i >= mix.shape[1]):
            arr = np.stack(batch_data, axis=0)
            B_padding = max((batch_size-len(batch_data)),0)
            arr = np.pad(arr,((0,B_padding),(0,0),(0,0)))

            # infer
            with mesh:
                arr = jnp.asarray(arr)
                x = model_apply(params,arr)
            
            # 动态调整窗口
            window = windowingArray.copy()
            if i - step == 0:  # First audio chunk, no fadein
                window[:fade_size] = 1
            elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                window[-fade_size:] = 1
            
            # 使用vmap优化的窗口化处理
            total_add_value = apply_windowing_vmap(x[..., :C], window)
            total_add_value = total_add_value[:batch_size-B_padding]
            total_add_value = np.asarray(total_add_value)
            
            # 批量处理结果累加
            for j in range(len(batch_locations)):
                start, l = batch_locations[j]
                result[..., start:start+l] += total_add_value[j][..., :l]
                counter[..., start:start+l] += window[..., :l]

            batch_data = []
            batch_locations = []

    estimated_sources = result / counter
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if length_init > 2 * border and (border > 0):
        # Remove pad
        estimated_sources = estimated_sources[..., border:-border]
    return estimated_sources

if __name__ == "__main__":
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
