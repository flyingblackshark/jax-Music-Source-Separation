# merge jax-bs-roformer and jax-mel-band-roformer
### Original https://github.com/ZFTurbo/Music-Source-Separation-Training

## Quick Start (Local)
```
pip install -e .
```

If you run from repo without installing:
```
PYTHONPATH=src python -m jaxmsst.infer --help
```

## Docker Build
```
git clone https://github.com/flyingblackshark/jax-Music-Source-Separation

sudo docker build --build-arg MODE=stable -f msst.Dockerfile -t jax-msst:dev .

docker save -o /bucket/msst/image.tar jax-msst:dev
```
## Docker Deploy
```
docker load -i /bucket/msst/image.tar

sudo docker run -d --net=host --privileged -v /bucket:/bucket jax-msst:dev python -m src.jaxmsst.webui --config_path /bucket/msst/model_options.yaml
```

## Inference (CLI)
Basic usage:
```
PYTHONPATH=src python -m jaxmsst.infer \
  --config_path configs/bs_roformer_logic.yaml \
  --start_check_point /path/to/model.ckpt \
  --input_folder ./input \
  --store_dir ./output
```

Notes:
- `configs/*.yaml` controls training/inference behavior (`train`, `data`, `data_loader`, `log`, `inference`).
- Model/audio params come from a model-side `config.yaml`. Provide it with `--model_config_path` or place `config.yaml` next to the checkpoint.
- On TPU/multi-device, set `inference.batch_size` to a multiple of device count.

Example: BS-Roformer HyperACE v2 vocals
```
python - <<'PY'
from pathlib import Path
import yaml

def to_list(obj):
    if isinstance(obj, tuple):
        return [to_list(item) for item in obj]
    if isinstance(obj, list):
        return [to_list(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_list(value) for key, value in obj.items()}
    return obj

source = Path('BS-Roformer-HyperACE/v2_voc/config.yaml')
data = yaml.load(source.read_text(), Loader=yaml.FullLoader)
data = to_list(data)
Path('BS-Roformer-HyperACE/v2_voc/config_omega.yaml').write_text(
    yaml.safe_dump(data, sort_keys=False)
)
PY

PYTHONPATH=src python -m jaxmsst.infer \
  --config_path configs/bs_roformer_logic.yaml \
  --model_config_path BS-Roformer-HyperACE/v2_voc/config_omega.yaml \
  --start_check_point BS-Roformer-HyperACE/v2_voc/bs_roformer_voc_hyperacev2.ckpt \
  --input_folder ./input \
  --store_dir ./output
```

## WebUI
```
PYTHONPATH=src python -m jaxmsst.webui --config_path configs/webui/model_options.yaml
```

## Training
```
PYTHONPATH=src python -m jaxmsst.train \
  --config configs/bs_roformer_base.yaml \
  --hardware gpu
```
