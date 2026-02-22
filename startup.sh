#!/bin/bash
set -e
exec > /tmp/startup.log 2>&1

cd /home/$(whoami)

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install optax tiktoken wandb huggingface_hub numpy

export WANDB_API_KEY="YOUR_WANDB_KEY_HERE"

git clone https://github.com/Kasties/nanogpt.git
cd nanogpt

python3 cached_fineweb10b.py
python3 train_gpt2.py