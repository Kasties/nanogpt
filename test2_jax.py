
### Changed Gelu to relu^2, Changed RoPE base to 1024, Added RMSNorm on embeddings


import jax
import jax.numpy as jnp
import optax
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import wandb
import time


# --- Hyperparameters ---
total_batch_size = 524288
batch_size = 256
block_size = 1024
grad_accum_steps = total_batch_size // (batch_size * block_size)
print(f"Using grad_accum_steps={grad_accum_steps} to achieve effective batch size of {total_batch_size}")
max_iters = 9536
vocab_size = 50304
eval_interval = 128
n_layer = 12
param_dtype = jnp.float32
compute_dtype = jnp.bfloat16
warmdown_steps = 2048
warmup_iters = 256
eval_batch_size = 16
learning_rate = 0.0018
weight_decay = 0.1
n_embd = 1024
num_heads = 4                                      
head_dim = n_embd // num_heads                     
attn_scale = 1.0 / (2 * n_layer) ** 0.5            # NEW: match PyTorch Block.attn_scale

wandb.init(project="nanogpt-jax")

rng = jax.random.PRNGKey(1337)
key, train_key = jax.random.split(rng)              # FIX: initialize train_key


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_bin_tokens(path: str):
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, f"magic number mismatch: {path}"
        assert header[1] == 1, f"unsupported shard version in: {path}"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, f"token count mismatch in {path}: header={ntok}, read={len(tokens)}"
    return tokens.astype(np.int32), ntok


def load_chunk(chunk_idx):
    path = f"fineweb10B/fineweb_train_{chunk_idx:06d}.bin"
    tokens, ntok = _read_bin_tokens(path)
    buf = np.zeros(100_000_000, dtype=np.int32)
    buf[:ntok] = tokens
    return jnp.array(buf, dtype=jnp.int32), jnp.array(ntok)


def load_val_chunk():
    path = "fineweb10B/fineweb_val_000000.bin"
    tokens, ntok = _read_bin_tokens(path)
    return jnp.array(tokens, dtype=jnp.int32), jnp.array(ntok)


val_data, val_n_tokens = load_val_chunk()


@jax.jit
def get_batch(data_arr, n_tokens, key):
    max_start = jnp.maximum(n_tokens - block_size - 1, 1)
    ix = jax.random.randint(key, (batch_size * grad_accum_steps,), 0, max_start)

    def get_slice(start_i):
        x = jax.lax.dynamic_slice(data_arr, (start_i,), (block_size,))
        y = jax.lax.dynamic_slice(data_arr, (start_i + 1,), (block_size,))
        return x, y

    x, y = jax.vmap(get_slice)(ix)
    return x, y


val_max_steps = 20  # number of eval microbatches, matches PyTorch default

@jax.jit
def get_eval_batch(data_arr, n_tokens, key):
    max_start = jnp.maximum(n_tokens - block_size - 1, 1)
    ix = jax.random.randint(key, (eval_batch_size,), 0, max_start)

    def get_slice(start_i):
        x = jax.lax.dynamic_slice(data_arr, (start_i,), (block_size,))
        y = jax.lax.dynamic_slice(data_arr, (start_i + 1,), (block_size,))
        return x, y

    x, y = jax.vmap(get_slice)(ix)
    return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def init_model_params(key, vocab_size, n_embd):
    keys = jax.random.split(key, num=6 * n_layer + 1)

    _offset = [0]
    def take(n=1):
        s = slice(_offset[0], _offset[0] + n)
        _offset[0] += n
        return keys[s] if n > 1 else keys[s][0]

    def p(x):
        return x.astype(param_dtype)

    # Weight tying: no separate lm_head — forward uses token_embedding.T
    params = {
        'token_embedding': p(jax.random.normal(take(), (vocab_size, n_embd)) * 0.02),
        'layers': {
            'W_q':            p(jax.vmap(lambda k: jax.random.normal(k, (n_embd, n_embd)) * 0.02)(take(n_layer))),
            'W_k':            p(jax.vmap(lambda k: jax.random.normal(k, (n_embd, n_embd)) * 0.02)(take(n_layer))),
            'W_v':            p(jax.vmap(lambda k: jax.random.normal(k, (n_embd, n_embd)) * 0.02)(take(n_layer))),
            'W_out':          p(jax.vmap(lambda k: jax.random.normal(k, (n_embd, n_embd)) * 0.02)(take(n_layer))),
            'W_ffwd':         p(jax.vmap(lambda k: jax.random.normal(k, (n_embd, 4 * n_embd)) * 0.02)(take(n_layer))),
            'W_ffwd_project': p(jax.vmap(lambda k: jax.random.normal(k, (4 * n_embd, n_embd)) * 0.02)(take(n_layer))),
        },
    }
    return params


def rms_norm(x, eps=1e-6):
    x_32 = x.astype(jnp.float32)
    norm = x_32 * jax.lax.rsqrt(jnp.mean(jnp.square(x_32), axis=-1, keepdims=True) + eps)
    return norm.astype(compute_dtype)


def rotary(seq_len):
    # ============================================================
    # CHANGE 2: RoPE base 1024 instead of 10000
    #   - For seq_len=1024, base=1024 concentrates frequencies better
    #   - Was: 10000.0
    # ============================================================
    inv_freq = 1.0 / (1024.0 ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)          # (T, head_dim//2)
    cos = jnp.cos(freqs)                    # (T, head_dim//2)
    sin = jnp.sin(freqs)                    # (T, head_dim//2)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """x: (B, T, H, D), cos/sin: (T, D//2)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[None, :, None, :]             # (1, T, 1, D//2)
    sin = sin[None, :, None, :]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1).astype(compute_dtype)


@jax.jit
def forward(params, idx):
    B, T = idx.shape

    x = params['token_embedding'][idx].astype(compute_dtype)     # (B, T, n_embd)

    # ============================================================
    # CHANGE 3: RMSNorm on embeddings right after lookup
    #   - Normalizes embedding scale before entering transformer
    #   - Was: nothing here
    # ============================================================
    x = rms_norm(x)

    # Precompute rotary tables once
    cos, sin = rotary(T)

    @jax.checkpoint
    def block_fn(x, layer_params):
        # ---------- Self-attention ----------
        x_norm = rms_norm(x)

        q = x_norm @ layer_params['W_q'].astype(compute_dtype)  # (B, T, n_embd)
        k = x_norm @ layer_params['W_k'].astype(compute_dtype)
        v = x_norm @ layer_params['W_v'].astype(compute_dtype)

        q = q.reshape(B, T, num_heads, head_dim)
        k = k.reshape(B, T, num_heads, head_dim)
        v = v.reshape(B, T, num_heads, head_dim)

        # RoPE — no QK-norm (matching PyTorch)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention  (B, T, H, D)
        attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.reshape(B, T, n_embd).astype(compute_dtype)
        attn_out = attn_out @ layer_params['W_out'].astype(compute_dtype)

        # Residual with attention scaling (matches PyTorch Block.attn_scale)
        x = x + attn_scale * attn_out

        # ---------- Feed-forward ----------
        x_norm = rms_norm(x)
        ffwd = x_norm @ layer_params['W_ffwd'].astype(compute_dtype)
        # ============================================================
        # CHANGE 1: Squared ReLU instead of GELU
        #   - Was: ffwd = jax.nn.gelu(ffwd, approximate=False)
        # ============================================================
        ffwd = jnp.square(jax.nn.relu(ffwd))
        ffwd = ffwd @ layer_params['W_ffwd_project'].astype(compute_dtype)
        x = x + ffwd                                           # no scaling on MLP (matches PyTorch)

        return x.astype(compute_dtype), None

    x, _ = jax.lax.scan(block_fn, x, params['layers'])

    # Final RMSNorm before projection (matches PyTorch `x = rmsnorm(x)` after blocks)
    x = rms_norm(x)

    # Weight-tied lm_head: logits = x @ token_embedding.T
    logits = x @ params['token_embedding'].T.astype(compute_dtype)

    return logits


@jax.jit
def loss_fn(params, idx, targets, key=None):
    logits = forward(params, idx)
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets
    ).mean()
    return loss


def eval_model(params, key):
    total_loss = 0.0
    for i in range(val_max_steps):
        key, eval_subkey = jax.random.split(key)
        xb, yb = get_eval_batch(val_data, val_n_tokens, eval_subkey)
        total_loss += float(loss_fn(params, xb, yb, key=eval_subkey))
    return total_loss / val_max_steps


# ---------------------------------------------------------------------------
# Optimizer — matches PyTorch's per-param grad norm + AdamW(betas=(0.9,0.95))
# ---------------------------------------------------------------------------

def unit_norm_grads():
    """Per-parameter gradient normalization: g / (||g|| + eps).
    Matches the PyTorch loop: `p.grad = p.grad / (p.grad.norm() + 1e-6)`."""
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        updates = jax.tree.map(
            lambda g: g / (jnp.linalg.norm(g) + 1e-6),
            updates,
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# LR schedule: linear warmup 0→lr, constant, linear warmdown lr→0
scheduler = optax.join_schedules(
    [
        optax.linear_schedule(0.0, learning_rate, warmup_iters),   # FIX: was lr→0
        optax.constant_schedule(learning_rate),
        optax.linear_schedule(learning_rate, 0.0, warmdown_steps),
    ],
    boundaries=[warmup_iters, max_iters - warmdown_steps],
)

optimizer = optax.chain(
    unit_norm_grads(),                                             # NEW: match PyTorch grad norm
    optax.adamw(learning_rate=scheduler, weight_decay=weight_decay,
                b1=0.9, b2=0.95),                                 # FIX: b2=0.95 to match PyTorch
)


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

@jax.jit
def train_step(params, opt_state, xb, yb, step, key):
    """Gradient-accumulating train step with scan over microbatches."""
    def scan_fn(carry, inputs):
        accloss, accgrad = carry
        x, y = inputs
        loss, grad = jax.value_and_grad(loss_fn)(params, x, y, key=key)
        new_loss = accloss + loss
        new_grad = jax.tree.map(lambda a, b: a + b, accgrad, grad)
        return (new_loss, new_grad), None

    zero_grads = jax.tree.map(jnp.zeros_like, params)
    (total_loss, total_grad), _ = jax.lax.scan(scan_fn, (0.0, zero_grads), (xb, yb))

    avg_loss = total_loss / grad_accum_steps
    avg_grad = jax.tree.map(lambda g: g / grad_accum_steps, total_grad)

    updates, new_opt_state = optimizer.update(avg_grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, avg_loss, new_opt_state


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

params = init_model_params(key, vocab_size, n_embd)
opt_state = optimizer.init(params)

devices = jax.devices()
print(f"Available devices: {devices}")
mesh = Mesh(devices, ('data',))
print(f"Using mesh: {mesh}")

num_chunks = 103
steps_per_chunk = 100_000_000 // (batch_size * block_size * grad_accum_steps)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

with mesh:
    params = jax.device_put(params, NamedSharding(mesh, P()))
    opt_state = jax.device_put(opt_state, NamedSharding(mesh, P()))

    for step in range(max_iters):
        # Load new data chunk when needed
        if step % steps_per_chunk == 0:
            chunk_idx = (step // steps_per_chunk) % num_chunks + 1
            current_chunk, current_chunk_n_tokens = load_chunk(chunk_idx)

        t0 = time.time()

        train_key, subkey = jax.random.split(train_key)
        xb, yb = get_batch(current_chunk, current_chunk_n_tokens, subkey)
        xb = xb.reshape(grad_accum_steps, -1, block_size)
        yb = yb.reshape(grad_accum_steps, -1, block_size)
        xb, yb = jax.device_put((xb, yb), NamedSharding(mesh, P(None, 'data', None)))

        params, avg_loss, opt_state = train_step(
            params, opt_state, xb, yb, jnp.array(step), subkey
        )

        avg_loss.block_until_ready()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (batch_size * block_size * grad_accum_steps) / (t1 - t0)

        log_dict = {
            "train/loss": float(avg_loss),
            "train/tokens_per_sec": tokens_per_sec,
            "train/learning_rate": float(scheduler(step)),
            "train/step_time_ms": dt,
        }

        if step % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            val_loss_float = eval_model(params, eval_key)
            log_dict["val/loss"] = val_loss_float
            print(
                f"Step {step}: train loss {float(avg_loss):.4f}, "
                f"val loss {val_loss_float:.4f} in {dt:.2f} ms, "
                f"tokens/sec: {tokens_per_sec:.2f}"
            )

        wandb.log(log_dict, step=step)