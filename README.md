# merge jax-bs-roformer and jax-mel-band-roformer
### Original https://github.com/ZFTurbo/Music-Source-Separation-Training


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
