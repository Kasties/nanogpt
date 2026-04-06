# nanoGPT Playground

This repository contains lightweight experiments for training and testing GPT-style language models in both PyTorch and JAX.

## What’s in this repo

- Training scripts (for example `train.py`, `train_gpt2.py`, and `fast_base_pytorch.py`).
- Small model/prototyping scripts (for example `bigram.py`, `v2.py`, and `a.py`).
- JAX test and comparison scripts (`test2_jax.py`, `test3_jax.py`, `test4_jax.py`, `jax_origianl.py`).
- Project/dependency configuration (`pyproject.toml`, `requirements.txt`, `uv.lock`).

## Quick start

1. Create a Python environment.
2. Install dependencies from `requirements.txt` or use `uv` with `pyproject.toml`.
3. Run a training script, e.g.:

```bash
python train.py
```

## Notes

- `input.txt` is used as local training text data for some scripts.
- Checkpoints/logs in this repo are for local experimentation.
