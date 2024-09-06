pip install --upgrade pip setuptools
pip install git+https://github.com/boris-kuz/jaxloudnorm
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
wget https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt