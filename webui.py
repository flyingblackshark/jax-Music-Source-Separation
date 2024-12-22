import librosa
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
import jax
import numpy as np
from jax.experimental.compilation_cache import compilation_cache as cc
from infer import load_model_from_config,demix_track
cc.set_cache_dir("./jax_cache")

MODEL_CONFIG_OPTIONS = {
    "BS Roformer去混响": ("configs/bs_roformer_base.yaml", "deverb_bs_roformer_8_256dim_8depth.ckpt"),
    "Melband Roformer提取人声": ("configs/mel_band_roformer_base.yaml", "MelBandRoformer.ckpt"),
    "Melband Roformer去混响": ("configs/mel_band_roformer_base.yaml", "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"), 
}

def run_folder(input_audio,model_config_name):
    config_path, model_path = MODEL_CONFIG_OPTIONS[model_config_name]
    model,params,hp = load_model_from_config(config_path,model_path)
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    mix, sr = librosa.load(input_audio, sr=44100, mono=False)
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    res = demix_track(model,params,mix,mesh,hp)
    res = np.asarray(res)
    estimates = res.squeeze(0)
    estimates_now = estimates.transpose(1,0)
    estimates_now = estimates_now
    return 44100,estimates_now


import gradio as gr
if __name__ == "__main__":

    
    jax.distributed.initialize()
    # 创建Gradio界面
    iface = gr.Interface(
        fn=run_folder,
        inputs=[
            gr.Audio(type="filepath"),
            gr.Dropdown(choices=list(MODEL_CONFIG_OPTIONS.keys()), label="模型配置组合"),
        ],
        outputs=gr.Audio(type="numpy"),             # 输出类型为音频文件
        title="人声提取",                          # 界面标题
        description="上传音频文件，输出提取后的人声" # 描述
    )
    iface.queue()
    iface.launch()
