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
batch_size = 512
block_size = 1024
grad_accum_steps = total_batch_size // (batch_size * block_size)
print(f"Using grad_accum_steps={grad_accum_steps} to achieve effective batch size of {total_batch_size}")
max_iters = 9536
vocab_size = 50304
eval_interval = 128
n_layer = 12
param_dtype = jnp.float32
compute_dtype = jnp.bfloat16
# ============================================================
# CHANGE 4: Warmdown = 40% of training (was fixed 2048 steps)
#   - Matches the Muon recipe from Doc 4
#   - No warmup needed with Muon (was 256 steps)
# ============================================================
warmdown_steps = int(max_iters * 0.4)
eval_batch_size = 16
n_embd = 1024
num_heads = 4
head_dim = n_embd // num_heads
attn_scale = 1.0 / (2 * n_layer) ** 0.5

wandb.init(project="nanogpt-jax")

rng = jax.random.PRNGKey(1337)
key, train_key = jax.random.split(rng)


# ---------------------------------------------------------------------------
# Data loading (unchanged)
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


val_max_steps = 20

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

    # ============================================================
    # CHANGE 5: Separate lm_head initialized to zeros (no weight tying)
    #   - Allows different learning rates for embed vs lm_head
    #   - Zero init means uniform predictions at start
    #   - Was: weight-tied (forward used token_embedding.T)
    # ============================================================
    params = {
        'token_embedding': p(jax.random.normal(take(), (vocab_size, n_embd)) * 0.02),
        'lm_head': jnp.zeros((n_embd, vocab_size), dtype=param_dtype),
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
    # CHANGE 2: RoPE base 1024
    inv_freq = 1.0 / (1024.0 ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1).astype(compute_dtype)


@jax.jit
def forward(params, idx):
    B, T = idx.shape

    x = params['token_embedding'][idx].astype(compute_dtype)

    # CHANGE 3: RMSNorm on embeddings
    x = rms_norm(x)

    cos, sin = rotary(T)

    @jax.checkpoint
    def block_fn(x, layer_params):
        # ---------- Self-attention ----------
        x_norm = rms_norm(x)

        q = x_norm @ layer_params['W_q'].astype(compute_dtype)
        k = x_norm @ layer_params['W_k'].astype(compute_dtype)
        v = x_norm @ layer_params['W_v'].astype(compute_dtype)

        q = q.reshape(B, T, num_heads, head_dim)
        k = k.reshape(B, T, num_heads, head_dim)
        v = v.reshape(B, T, num_heads, head_dim)

        # ============================================================
        # CHANGE 7: QK-norm — stabilizes attention with high Muon LRs
        #   - Without this, Q/K magnitudes grow unchecked → attention
        #     logits explode → training diverges after ~2k steps
        #   - Was: no normalization before RoPE
        # ============================================================
        q = rms_norm(q)
        k = rms_norm(k)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.reshape(B, T, n_embd).astype(compute_dtype)
        attn_out = attn_out @ layer_params['W_out'].astype(compute_dtype)

        x = x + attn_scale * attn_out

        # ---------- Feed-forward ----------
        x_norm = rms_norm(x)
        ffwd = x_norm @ layer_params['W_ffwd'].astype(compute_dtype)
        # CHANGE 1: Squared ReLU
        ffwd = jnp.square(jax.nn.relu(ffwd))
        ffwd = ffwd @ layer_params['W_ffwd_project'].astype(compute_dtype)
        x = x + ffwd

        return x.astype(compute_dtype), None

    x, _ = jax.lax.scan(block_fn, x, params['layers'])

    x = rms_norm(x)

    # ============================================================
    # CHANGE 5b: Use separate lm_head instead of token_embedding.T
    #   - Was: logits = x @ params['token_embedding'].T
    # ============================================================
    logits = x @ params['lm_head'].astype(compute_dtype)

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
# CHANGE 6: Muon optimizer + multi-optimizer
#   - Was: single AdamW with unit_norm_grads for all params
#   - Now: Muon (Newton-Schulz) for layer weight matrices
#          Adam lr=0.6 for token_embedding
#          Adam lr=0.008 for lm_head
#   - This is the single biggest convergence improvement
# ---------------------------------------------------------------------------

def newton_schulz(X, steps=5, eps=1e-7):
    """Approximate the matrix sign function via Newton-Schulz iteration.
    This orthogonalizes the update direction, which is the key idea behind Muon."""
    assert len(X.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    def _update_loop(X):
        X = X.astype(compute_dtype)
        X = X / (jnp.linalg.norm(X) + eps)
        def body(_, X):
            A = X @ X.T
            B = b * A + c * A @ A
            return a * X + B @ X
        return jax.lax.fori_loop(0, steps, body, X)

    def tall_case(X):
        # X.shape[0] > X.shape[1]: transpose, run, transpose back
        return _update_loop(X.T).T.astype(X.dtype)

    def wide_case(X):
        # X.shape[0] <= X.shape[1]: run directly
        return _update_loop(X).astype(X.dtype)

    return jax.lax.cond(X.shape[0] > X.shape[1], tall_case, wide_case, X)


def muon_apply(grad_layers, muon_buffers, step):
    """Apply Muon optimizer to all layer weight matrices.

    Muon = SGD with momentum + Newton-Schulz orthogonalization.
    The orthogonalization makes each update direction approximately orthogonal,
    which empirically gives much faster convergence than AdamW for weight matrices.
    """
    # Momentum warmup: 0.85 → 0.95 over first 500 steps
    frac = jnp.minimum(step / 500.0, 1.0)
    momentum = (1 - frac) * 0.85 + frac * 0.95

    # LR schedule: constant 0.04, then linear warmdown to 0
    lr = muon_lr_schedule(step)

    def update_one(g, buf):
        # Momentum update
        new_buf = momentum * buf + g
        # Nesterov lookahead
        direction = g + momentum * new_buf
        # Newton-Schulz orthogonalization (vmap over layer dimension)
        ortho = jax.vmap(newton_schulz)(direction)
        # Scale by sqrt(fan_in / fan_out) when fan_in > fan_out
        scale = jnp.sqrt(jnp.maximum(1.0, g.shape[-2] / g.shape[-1]))
        return -lr * ortho * scale, new_buf

    updates = {}
    new_buffers = {}
    for k in grad_layers:
        updates[k], new_buffers[k] = update_one(grad_layers[k], muon_buffers[k])

    return updates, new_buffers


# --- LR schedules ---
# Muon: constant 0.04, then linear decay to 0 over last 40%
muon_lr_schedule = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0.04),
        optax.linear_schedule(0.04, 0.0, warmdown_steps),
    ],
    boundaries=[max_iters - warmdown_steps],
)

# Adam for embeddings: constant 0.6, then linear decay to 0
scheduler_embed = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0.6),
        optax.linear_schedule(0.6, 0.0, warmdown_steps),
    ],
    boundaries=[max_iters - warmdown_steps],
)

# Adam for lm_head: constant 0.008, then linear decay to 0
scheduler_lm = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0.008),
        optax.linear_schedule(0.008, 0.0, warmdown_steps),
    ],
    boundaries=[max_iters - warmdown_steps],
)

# Optax Adam instances (no weight decay — Muon handles layer weights separately)
embed_opt = optax.adam(learning_rate=scheduler_embed, b1=0.9, b2=0.95, eps=1e-10)
lm_opt = optax.adam(learning_rate=scheduler_lm, b1=0.9, b2=0.95, eps=1e-10)


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

@jax.jit
def train_step(params, muon_buffers, embed_state, lm_state, xb, yb, step, key):
    """Gradient-accumulating train step with three-way optimizer split:
    - Muon for layers (weight matrices)
    - Adam lr=0.6 for token_embedding
    - Adam lr=0.008 for lm_head
    """
    # --- Gradient accumulation via scan ---
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

    # --- Muon for layer weights ---
    layer_updates, new_buffers = muon_apply(avg_grad['layers'], muon_buffers, step)
    new_layers = jax.tree.map(lambda p, u: p + u, params['layers'], layer_updates)

    # --- Adam for token_embedding ---
    embed_updates, new_embed_state = embed_opt.update(
        avg_grad['token_embedding'], embed_state, params['token_embedding']
    )
    new_embed = optax.apply_updates(params['token_embedding'], embed_updates)

    # --- Adam for lm_head ---
    lm_updates, new_lm_state = lm_opt.update(
        avg_grad['lm_head'], lm_state, params['lm_head']
    )
    new_lm = optax.apply_updates(params['lm_head'], lm_updates)

    new_params = {
        'token_embedding': new_embed,
        'lm_head': new_lm,
        'layers': new_layers,
    }

    return new_params, new_buffers, new_embed_state, new_lm_state, avg_loss


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

params = init_model_params(key, vocab_size, n_embd)

# Muon momentum buffers (one per layer weight, same shape, all zeros)
muon_buffers = jax.tree.map(jnp.zeros_like, params['layers'])

# Adam states for embedding and lm_head
embed_adam_state = embed_opt.init(params['token_embedding'])
lm_adam_state = lm_opt.init(params['lm_head'])

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
    muon_buffers = jax.device_put(muon_buffers, NamedSharding(mesh, P()))
    embed_adam_state = jax.device_put(embed_adam_state, NamedSharding(mesh, P()))
    lm_adam_state = jax.device_put(lm_adam_state, NamedSharding(mesh, P()))

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

        params, muon_buffers, embed_adam_state, lm_adam_state, avg_loss = train_step(
            params, muon_buffers, embed_adam_state, lm_adam_state,
            xb, yb, jnp.array(step), subkey
        )

        avg_loss.block_until_ready()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (batch_size * block_size * grad_accum_steps) / (t1 - t0)

        log_dict = {
            "train/loss": float(avg_loss),
            "train/tokens_per_sec": tokens_per_sec,
            "train/lr_embed": float(scheduler_embed(step)),
            "train/lr_lm": float(scheduler_lm(step)),
            "train/lr_muon": float(muon_lr_schedule(step)),
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