pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
wget https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_256dim_8depth.ckpt