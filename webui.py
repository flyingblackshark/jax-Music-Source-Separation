import argparse
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
import os
from omegaconf import OmegaConf
from functools import partial
cc.set_cache_dir("./jax_cache")

# 从配置文件加载模型选项
def load_model_config_options(config_path):
    config = OmegaConf.load(config_path)
    model_options = {}
    for name, options in config.model_options.items():
        model_options[name] = (options.config_path, options.model_path)
    return model_options

def run_folder(input_audio,model_config_name,configs):
    config_path, model_path = configs[model_config_name]
    model,params,hp = load_model_from_config(config_path,model_path)
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    mix, sr = librosa.load(input_audio, sr=44100, mono=False)
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    res = demix_track(model,params,mix,mesh,hp)
    res = np.asarray(res)
    
    # 根据config中instruments数量返回对应数量的音频
    instruments = hp.model.instruments
    outputs = []
    for i, instrument in enumerate(instruments):
        estimate = res[i].transpose(1,0)
        outputs.append((44100, estimate))
    
    return outputs


import gradio as gr
import librosa

def initialize_jax_for_gpu():
  """Jax distribute initialize for GPUs."""
  if os.environ.get("JAX_COORDINATOR_IP") is not None:
    coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
    coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_ip}:{coordinator_port}",
        num_processes=int(os.getenv("NNODES")),
        process_id=int(os.getenv("NODE_RANK")),
    )


def maybe_initialize_jax_distributed_system():

  if os.getenv("GPU_MODE") is not None:
    initialize_jax_for_gpu()
  else:
    jax.distributed.initialize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.getenv('CONFIG_PATH', 'configs/webui/model_options.yaml'),
                        help="path to config file")
    args = parser.parse_args()
    maybe_initialize_jax_distributed_system()
    configs = load_model_config_options(args.config_path)
    # 动态创建输出组件，根据第一个模型配置的instruments数量
    first_config = list(configs.keys())[0]
    config_path, _ = configs[first_config]
    hp = OmegaConf.load(config_path)
    instruments = hp.model.instruments
    
    # 创建对应数量的音频输出组件
    outputs = [gr.Audio(type="numpy", format="mp3",label=f"{instrument}") for instrument in instruments]
    
    # 创建Gradio界面
    iface = gr.Interface(
        fn=partial(run_folder,configs=configs),
        inputs=[
            gr.Audio(type="filepath"),
            gr.Dropdown(choices=list(configs.keys()), label="模型配置组合"),
        ],
        outputs=outputs,
        title="音乐源分离",
        description=f"上传音频文件，输出分离后的{len(instruments)}种乐器: {', '.join(instruments)}"
    )
    iface.queue()
    iface.launch(server_name="0.0.0.0")
