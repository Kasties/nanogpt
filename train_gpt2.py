import jax
import jax.numpy as jnp
import optax
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import wandb
import pickle
import time
from jax.ad_checkpoint import checkpoint as remat
# --- TPU Detection and Hardware-Aware Config ---
def get_tpu_type():
    """Detect TPU version from device kind string."""
    devices = jax.devices()
    if not devices or devices[0].platform != 'tpu':
        return None
    kind = devices[0].device_kind.lower()
    if 'v6e' in kind or 'trillium' in kind or 'v6' in kind:
        return 'v6e'
    if 'v5e' in kind or 'v5litepod' in kind:
        return 'v5e'
    if 'v5p' in kind or 'v5' in kind:
        return 'v5p'
    if 'v4' in kind:
        return 'v4'
    if 'v3' in kind:
        return 'v3'
    if 'v2' in kind:
        return 'v2'
    return 'unknown'

tpu_type = get_tpu_type()
print(f"Detected TPU type: {tpu_type}")

# MXU sizes:
#   v6e/Trillium: 256x256 -> d_head=256, num_heads=4, n_embd=1024
#   v2/v3/v4/v5e/v5p: 128x128 -> d_head=128, num_heads=6, n_embd=768
if tpu_type in ('v6e',):
    n_embd = 1024
    num_heads = 4
    print(f"Using v6e-optimized config: n_embd={n_embd}, num_heads={num_heads}, d_head={n_embd // num_heads}")
else:
    n_embd = 768
    num_heads = 6
    print(f"Using standard config: n_embd={n_embd}, num_heads={num_heads}, d_head={n_embd // num_heads}")

# --- Hyperparameters ---
total_batch_size = 524288
batch_size = 64
block_size = 1024
grad_accum_steps = total_batch_size // (batch_size*block_size)
print(f"Using grad_accum_steps={grad_accum_steps} to achieve effective batch size of {total_batch_size}")
max_iters = 4000
device = 'tpu' 
vocab_size = 50304 
eval_interval = 100
n_layer = 12
param_dtype = jnp.bfloat16
compute_dtype = jnp.bfloat16
warmdown_steps = int(max_iters * 0.4)
eval_batch_size = 16
wandb.init(project="nanogpt-jax")

rng = jax.random.PRNGKey(1337)
key, subkey = jax.random.split(rng)


data_buffer_train = jnp.zeros((100_000_000,), dtype=jnp.int32)

def _read_bin_tokens(path: str):
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, f"magic number mismatch: {path}"
        assert header[1] == 1, f"unsupported shard version in: {path}"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, f"token count mismatch in {path}: header={ntok}, read={len(tokens)}"
    return tokens.astype(np.int32), ntok

def load_chunk(chunk_idx, data_buffer=data_buffer_train):
    path = f"fineweb10B/fineweb_train_{chunk_idx:06d}.bin"
    tokens, ntok = _read_bin_tokens(path)
    tokens_jax = jnp.array(tokens, dtype=jnp.int32)
    # write only valid prefix; sampling will be limited by ntok
    data_buffer = data_buffer.at[:ntok].set(tokens_jax)
    return data_buffer, ntok

def load_val_chunk():
    path = "fineweb10B/fineweb_val_000000.bin"
    tokens, ntok = _read_bin_tokens(path)
    return jnp.array(tokens, dtype=jnp.int32), ntok

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

@jax.jit
def get_eval_batch(data_arr, n_tokens, key):
    max_start = jnp.maximum(n_tokens - block_size - 1, 1)
    ix = jax.random.randint(key, (eval_batch_size * grad_accum_steps,), 0, max_start)

    def get_slice(start_i):
        x = jax.lax.dynamic_slice(data_arr, (start_i,), (block_size,))
        y = jax.lax.dynamic_slice(data_arr, (start_i + 1,), (block_size,))
        return x, y

    x, y = jax.vmap(get_slice)(ix)
    return x, y
def init_model_params(key, vocab_size, n_embd):
    head_size = n_embd // num_heads
    keys = jax.random.split(key, num=6 * n_layer + 2)
    
    def p(x):
        return x.astype(param_dtype)

    # Small helper to grab the next n keys from the pool
    _offset = [0]
    def take(n=1):
        s = slice(_offset[0], _offset[0] + n)
        _offset[0] += n
        return keys[s] if n > 1 else keys[s][0]
    params = {
        'token_embedding':      p(jax.random.normal(take(), (vocab_size, n_embd)) * 0.02),
        'lm_head':            jnp.zeros((n_embd, vocab_size), dtype=param_dtype),
        'layers': {
            'W_q':            p(jax.vmap(lambda k: jax.random.normal(k, (num_heads, n_embd, head_size)) * 0.02)(take(n_layer))),
            'W_k':            p(jax.vmap(lambda k: jax.random.normal(k, (num_heads, n_embd, head_size)) * 0.02)(take(n_layer))),
            'W_v':            p(jax.vmap(lambda k: jax.random.normal(k, (num_heads, n_embd, head_size)) * 0.02)(take(n_layer))),
            'W_ffwd':         p(jax.vmap(lambda k: jax.random.normal(k, (n_embd, 4 * n_embd)) * 0.02)(take(n_layer))),
            'W_out': jnp.zeros((n_layer,num_heads*head_size,n_embd), dtype=param_dtype),
            'W_ffwd_project': jnp.zeros((n_layer,4*n_embd,n_embd), dtype=param_dtype),
            'lamb' : jnp.full((n_layer,), 0.5, dtype=param_dtype)
            },
    }       
    return params

def newton_schulz(X, steps=5, eps=1e-7):
    if X.ndim < 2:
        return X
    X = X.astype(compute_dtype)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = X / (jnp.linalg.norm(X) + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    def body(_, X):
        A = X @ X.T
        B = b * A + c * A @ A
        return a * X + B @ X
    X = jax.lax.fori_loop(0, steps, body, X)
    if transposed:
        X = X.T
    return X

def muon_update(grad, momentum_buf, momentum=0.95):
    buf = momentum * momentum_buf + grad
    g = grad + momentum * buf  # nesterov
    g = jax.vmap(newton_schulz)(g)  # vmap over layer dim
    g = g * jnp.sqrt(jnp.maximum(1.0, g.shape[-2] / g.shape[-1]))
    return g, buf


def rotary(x):
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (1024 ** (jnp.arange(0, head_dim, 2) / head_dim))
    seq_len = x.shape[1]
    t = jnp.arange(seq_len)
    freqs = jnp.einsum('i,j->ij', t, inv_freq)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin



@jax.jit
def forward(params, idx):
    def scan_fn(x, layer_params):
        x,_  = remat(transformer_block)(x,layer_params, v1)
        return x , None

    def apply_rotary_emb(x,cos, sin):
        d = x.shape[3]//2
        x1, x2 = x[:,:,:,:d], x[:,:,:,d:]
        sin = sin[:, None, :]
        cos = cos[:, None, :]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * -sin + x2 * cos
        return jnp.concatenate([y1.astype(compute_dtype), y2.astype(compute_dtype)], axis=-1)

    def rms_norm(x, eps=1e-5):
        x_32 = x.astype(param_dtype)
        norm = jnp.sqrt(jnp.mean(jnp.square(x_32), axis=-1, keepdims=True) + eps)
        out = x_32 / norm
        return out.astype(compute_dtype)


    def feedforward(x, layer_params):
        out = x @ layer_params['W_ffwd']# (B, T, n_embd) @ (n_embd, 4*n_embd) -> (B, T, 4*n _embd)
        out = jnp.square(jax.nn.relu(out.astype(compute_dtype))) # (B, T, n_embd) @ (n_embd, 4*n_embd) -> (B, T, 4*n _embd) use relu squared
        out = out @ layer_params['W_ffwd_project'] # (B, T, 4*n_embd) @ (4*n_embd, n_embd) -> (B, T, n_embd)
        return out

    def multi_head_attention(x, layer_params, v1=None):
        q = jnp.einsum('bte,hes->bths', x, layer_params['W_q'].astype(compute_dtype))
        k = jnp.einsum('bte,hes->bths', x, layer_params['W_k'].astype(compute_dtype))
        v = jnp.einsum('bte,hes->bths', x, layer_params['W_v'].astype(compute_dtype))

        # value residual mixing (no-op when v1 is None, i.e. first layer)
        if v1 is not None:
            lamb = layer_params['lamb'].astype(compute_dtype)
            v = (1 - lamb) * v + lamb * v1

        cos, sin = rotary(q)
        q, k = rms_norm(q), rms_norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        out = out.astype(compute_dtype).reshape(B, T, -1)
        out = out @ layer_params['W_out']
        return out, v

    def transformer_block(x, layer_params, v1=None):
        out, v = multi_head_attention(rms_norm(x), layer_params, v1)
        x = x + out
        x = x + feedforward(rms_norm(x), layer_params)
        return x.astype(compute_dtype), v
    B,T = idx.shape


    x = params['token_embedding'][idx] # (B, T, n_embd)

    x = rms_norm(x) # (B, T, n_embd)
    x, v1 = transformer_block(x, jax.tree.map(lambda p: p[0], params['layers']), v1=None)

    x, _ = jax.lax.scan(scan_fn,x, jax.tree.map(lambda p: p[1:], params['layers'])) # scan over layers
    x = rms_norm(x) # (B, T, n_embd)
    # x = x.astype(param_dtype) @ params['lm_head'] # (B, T, n_embd) @ (n_embd, vocab_size) -> (B, T, vocab_size)
    # x = 15 * jnp.tanh(x / 15)
    return x # (B, T, vocab_size)


def chunked_loss(logits_fn, targets, n_chunks=8):
    """Compute cross-entropy without materializing full (B*T, vocab) tensor."""
    B, T = targets.shape
    targets = targets.reshape(-1)
    chunk_size = (B * T) // n_chunks
    total_loss = 0.0
    for i in range(n_chunks):
        s = i * chunk_size
        logit_chunk = logits_fn(s, chunk_size)  # only compute this slice
        loss_chunk = optax.softmax_cross_entropy_with_integer_labels(
            logit_chunk.astype(param_dtype), targets[s:s+chunk_size]
        ).sum()
        total_loss += loss_chunk
    return total_loss / (B * T)

@jax.jit
def loss_fn(params, idx, targets, key=None):
    hidden = forward(params, idx)  # (B, T, n_embd)
    def logits_fn(start, size):
        h_chunk = hidden.reshape(-1, n_embd)[start:start+size]
        return h_chunk @ params['lm_head']  # (chunk, vocab)
    return chunked_loss(logits_fn, targets, n_chunks=8)

def muon_apply(grad_layers, muon_buffers, step):
    frac = jnp.minimum(step / 500.0, 1.0)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    lr = muon_lr_schedule(step)

    def update_one(g, buf):
        new_buf = momentum * buf + g
        direction = g + momentum * new_buf
        # orthogonalize — pick vmap depth based on shape
        if g.ndim == 4:  # (n_layer, num_heads, n_embd, head_size)
            ortho = jax.vmap(jax.vmap(newton_schulz))(direction)
            scale = jnp.sqrt(jnp.maximum(1.0, g.shape[-3] / g.shape[-1]))
            return -lr * ortho * scale, new_buf
        elif g.ndim == 3:  # (n_layer, dim1, dim2)
            ortho = jax.vmap(newton_schulz)(direction)
            scale = jnp.sqrt(jnp.maximum(1.0, g.shape[-2] / g.shape[-1]))
            return -lr * ortho * scale, new_buf
        elif g.ndim == 2:  # (dim1, dim2)
            ortho = newton_schulz(direction)
            scale = jnp.sqrt(jnp.maximum(1.0, g.shape[-2] / g.shape[-1]))
            return -lr * ortho * scale, new_buf
        else:
            return -lr * direction, new_buf

    updates = {}
    new_buffers = {}
    for k in grad_layers:
        updates[k], new_buffers[k] = update_one(grad_layers[k], muon_buffers[k])

    return updates, new_buffers

@jax.jit
def train_step(params, muon_buffers, embed_state, lm_state, xb, yb, step, key):
    def scan_fn(carry, inputs):
        accloss, accgrad = carry
        x, y = inputs
        loss, grad = jax.value_and_grad(loss_fn)(params, x, y, key=key)
        new_loss = accloss + loss
        new_grad = jax.tree.map(lambda a, b: a + b, accgrad, grad)
        return (new_loss, new_grad), None
    (losses, grads), _  = jax.lax.scan(scan_fn, (0.0, jax.tree.map(jnp.zeros_like, params)), (xb, yb))
    avg_loss = losses/grad_accum_steps
    avg_grad = jax.tree.map(lambda g: g / grad_accum_steps, grads)

    # Muon for layer weights
    layer_updates, new_buffers = muon_apply(avg_grad['layers'], muon_buffers, step)
    new_layers = jax.tree.map(lambda p, u: p + u, params['layers'], layer_updates)

    # Adam for embedding (lr=0.6)
    embed_updates, new_embed_state = embed_opt.update(
        avg_grad['token_embedding'], embed_state, params['token_embedding']
    )
    new_embed = optax.apply_updates(params['token_embedding'], embed_updates)

    # Adam for lm_head (lr=0.008)
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

def eval_model(params, key):
    xb, yb = get_eval_batch(val_data, val_n_tokens, key)
    val_loss = loss_fn(params, xb, yb, key=key)
    return val_loss

# --- Training Loop ---
params = init_model_params(subkey, vocab_size, n_embd)

print(f"Training on {device}...")
print(f"Config: n_embd={n_embd}, num_heads={num_heads}, d_head={n_embd // num_heads}, n_layer={n_layer}")
key, train_key = jax.random.split(key)



# --- Init momentum buffers for Muon params ---
muon_buffers = jax.tree.map(jnp.zeros_like, params['layers'])

# --- Adam only handles embeddings/lm_head ---
adam_params = {
    'token_embedding': params['token_embedding'],
    'lm_head': params['lm_head'],
}

scheduler_embed = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0.6),
        optax.linear_schedule(0.6, 0.0, warmdown_steps)
    ],
    boundaries=[max_iters - warmdown_steps]
)

embed_opt = optax.adam(learning_rate=scheduler_embed, b1=0.9, b2=0.95, eps=1e-10)
embed_adam_state = embed_opt.init(params['token_embedding'])
scheduler_lm = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0.008),
        optax.linear_schedule(0.008, 0.0, warmdown_steps)
    ],
    boundaries=[max_iters - warmdown_steps]
)
lm_opt = optax.adam(learning_rate=scheduler_lm, b1=0.9, b2=0.95, eps=1e-10)
lm_adam_state = lm_opt.init(params['lm_head'])
# --- Muon LR schedule (same shape: constant then warmdown) ---
muon_lr_schedule = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0.04),
        optax.linear_schedule(0.04, 0.0, warmdown_steps)
    ],
    boundaries=[max_iters - warmdown_steps]
)


devices = jax.devices()
print(f"Available devices: {devices}")
mesh = Mesh(devices, ('data',))
print(f"Using mesh: {mesh}")

num_chunks = 103  # full fineweb10B. Each chunk is 100M tokens
steps_per_chunk = (100_000_000) // (batch_size * block_size * grad_accum_steps)
with mesh:
    params = jax.device_put(params, NamedSharding(mesh, P()))
    muon_buffers = jax.device_put(muon_buffers, NamedSharding(mesh, P()))
    embed_adam_state = jax.device_put(embed_adam_state, NamedSharding(mesh, P()))
    lm_adam_state = jax.device_put(lm_adam_state, NamedSharding(mesh, P()))
    for step in range(max_iters):
        if step % steps_per_chunk == 0:
            chunk_idx = (step // steps_per_chunk) % num_chunks + 1
            current_chunk, current_chunk_n_tokens = load_chunk(chunk_idx)
        t0 = time.time()

        train_key, subkey = jax.random.split(train_key)
        xb, yb = get_batch(current_chunk, current_chunk_n_tokens, subkey)
        xb = xb.reshape(grad_accum_steps, -1, block_size)
        yb = yb.reshape(grad_accum_steps, -1, block_size)
        xb, yb = jax.device_put((xb, yb), NamedSharding(mesh, P(None,'data', None)))

        params, muon_buffers, embed_adam_state, lm_adam_state, avg_loss = train_step(
            params, muon_buffers, embed_adam_state, lm_adam_state, xb, yb, jnp.array(step), subkey
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
            val_loss = eval_model(params, eval_key)
            val_loss_float = float(val_loss)
            log_dict["val/loss"] = val_loss_float
            print(f"Step {step}: train loss {float(avg_loss):.4f}, val loss {val_loss_float:.4f} in {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")

        wandb.log(log_dict, step=step)

# write params to disk

print("Saving model parameters...")
# 1. Move parameters to CPU
# 2. Convert to standard numpy float32 (NumPy doesn't natively support bfloat16)
params_cpu = jax.tree.map(lambda x: np.array(x, dtype=np.float32), params)

with open("gpt2_params.pkl", "wb") as f:
    pickle.dump(params_cpu, f)
    
print("Model saved successfully as gpt2_params.pkl!")