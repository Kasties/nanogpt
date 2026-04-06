# nanoGPT Playground

This repository is a personal sandbox for understanding how GPT-style language models are built, trained, and evaluated from scratch. The code focuses on small, readable experiments so you can follow the full workflow: preparing text data, defining a transformer model, training it, and sampling generated text.

## Project explanation

The goal of this project is learning-by-building. Instead of relying on a large framework, the scripts in this repo break the model pipeline into simple pieces so it is easier to inspect and modify:

- data loading and token batching
- model architecture experiments
- training loop and optimization behavior
- checkpointing and quick evaluation/sampling

This makes the repo useful for tinkering with hyperparameters, testing implementation ideas, and comparing variants of the same core GPT approach.

## What’s in this repo

- Training scripts (for example `train.py` and `train_gpt2.py`).
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
