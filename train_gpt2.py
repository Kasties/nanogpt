import jax
import jax.numpy as jnp
import optax
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import os
import wandb
import pickle
import numpy as np
import tiktoken
import time
import pickle
import numpy as np
# --- Hyperparameters ---
total_batch_size = 524288
batch_size = 64 * 2
block_size = 1024
grad_accum_steps = total_batch_size // (batch_size*block_size)
print(f"Using grad_accum_steps={grad_accum_steps} to achieve effective batch size of {total_batch_size}")
max_iters = 10000
learning_rate = 3e-4
device = 'tpu' 
vocab_size = 50304 
eval_interval = 300
n_embd = 768
n_layer = 12
dropout = 0.0
num_heads = 12
param_dtype = jnp.float32
compute_dtype = jnp.bfloat16
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 1000
eval_batch_size = 16
# tril = jnp.tril(jnp.ones((block_size, block_size), dtype=bool))
wandb.init(project="nanogpt-jax")

rng = jax.random.PRNGKey(1337)
key, subkey = jax.random.split(rng)

import tiktoken

enc = tiktoken.get_encoding("gpt2")
def load_chunk(chunk_idx):
    tokens = np.fromfile(f"fineweb10B/fineweb_train_{chunk_idx:06d}.bin", dtype=np.uint16)
    return jnp.array(tokens.astype(np.int32))
val_data = jnp.array(np.fromfile("fineweb10B/fineweb_val_000000.bin", dtype=np.uint16).astype(np.int32))
@jax.jit
def get_batch(data_arr, key):
    
    # Generate random starting indices
    ix = jax.random.randint(key, (batch_size,), 0, len(data_arr) - block_size - 1)
    
    # Function to grab one slice, given one start index
    def get_slice(start_i):
        x = jax.lax.dynamic_slice(data_arr, (start_i,), (block_size,))
        y = jax.lax.dynamic_slice(data_arr, (start_i + 1,), (block_size,))
        return x, y
        
    # Vectorize this over the batch of indices
    x, y = jax.vmap(get_slice)(ix)
    return x, y
@jax.jit
def get_eval_batch(data_arr, key):
    ix = jax.random.randint(key, (eval_batch_size,), 0, len(data_arr) - block_size - 1)
    def get_slice(start_i):
        x = jax.lax.dynamic_slice(data_arr, (start_i,), (block_size,))
        y = jax.lax.dynamic_slice(data_arr, (start_i + 1,), (block_size,))
        return x, y
    x, y = jax.vmap(get_slice)(ix)
    return x, y

def init_model_params(key, vocab_size, n_embd):
    
    head_size = n_embd // num_heads
    keys = jax.random.split(key, num=10*n_layer + 2) # We need a lot of random keys, so we split the key into many subkeys at once

    def p(x):
        return x.astype(param_dtype)

    params = {
        'token_embedding': p(jax.random.normal(keys[0], (vocab_size, n_embd)) * 0.02),
        'positional_embedding': p(jax.random.normal(keys[1], (block_size, n_embd)) * 0.01),
        'W_q': [p(jax.random.normal(keys[2+i], (num_heads, n_embd, head_size)) * 0.02) for i in range(n_layer)],
        'W_k': [p(jax.random.normal(keys[2+n_layer+i], (num_heads, n_embd, head_size)) * 0.02) for i in range(n_layer)],
        'W_v': [p(jax.random.normal(keys[2+2*n_layer+i], (num_heads, n_embd, head_size)) * 0.02) for i in range(n_layer)],
        'W_out': [p(jax.random.normal(keys[2+3*n_layer+i], (num_heads * head_size, n_embd)) * (0.02 / jnp.sqrt(2*n_layer))) for i in range(n_layer)],
        'W_ffwd': [p(jax.random.normal(keys[2+4*n_layer+i], (n_embd, 4*n_embd)) * 0.02) for i in range(n_layer)],
        'W_ffwd_project': [p(jax.random.normal(keys[2+5*n_layer+i], (4*n_embd, n_embd)) * (0.02 / jnp.sqrt(2*n_layer))) for i in range(n_layer)],
        'ln1_gamma': [jnp.ones((n_embd,), dtype=param_dtype) for _ in range(n_layer)],
        'ln1_beta':  [jnp.zeros((n_embd,), dtype=param_dtype) for _ in range(n_layer)],
        'ln2_gamma': [jnp.ones((n_embd,), dtype=param_dtype) for _ in range(n_layer)],
        'ln2_beta':  [jnp.zeros((n_embd,), dtype=param_dtype) for _ in range(n_layer)],
        'ln_f_gamma': jnp.ones((n_embd,), dtype=param_dtype),
        'ln_f_beta':  jnp.zeros((n_embd,), dtype=param_dtype),
    }
    return params

@partial(jax.jit, static_argnames=['is_training'])
def forward(params, idx, is_training=False, target=None, key=None):
    def layer_norm(x,gamma,beta,eps=1e-5):
        x = x.astype(jnp.float32) 
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma.astype(jnp.float32) * (x - mean) / jnp.sqrt(var + eps) + beta.astype(jnp.float32)
    def apply_dropout(x, rate=dropout, key=None, is_training=False):
        if rate == 0.0 or not is_training or key is None:
            return x
        keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
        return x * keep / (1.0 - rate)

    def feedforward(x, layer_idx):
        out = x @ params['W_ffwd'][layer_idx]# (B, T, n_embd) @ (n_embd, 4*n_embd) -> (B, T, 4*n _embd)
        out = jax.nn.gelu(out.astype(jnp.float32),approximate=True).astype(compute_dtype) # (B, T, n_embd) @ (n_embd, 4*n_embd) -> (B, T, 4*n _embd) use aproximate gelu as it was the original activation in GPT2
        out = out @ params['W_ffwd_project'][layer_idx] # (B, T, 4*n_embd) @ (4*n_embd, n_embd) -> (B, T, n_embd)
        # out = apply_dropout(out, key=jax.random.PRNGKey(42), is_training=is_training) # No dropout for simplicity
        return out.astype(jnp.float32)

    def multi_head_attention(x, layer_idx, key=None, is_training=False):
            # (B, T, n_embd) -> (B, T, num_heads * head_size)
            q = jnp.einsum('bte,hes->bths', x, params['W_q'][layer_idx]) # (B, T, n_embd) @ (n_embd, head_size) -> (B, T, head_size)
            k = jnp.einsum('bte,hes->bths', x, params['W_k'][layer_idx]) # (B, T, n_embd) @ (n_embd, head_size) -> (B, T, head_size)
            v = jnp.einsum('bte,hes->bths', x, params['W_v'][layer_idx]) # (B, T, n_embd) @ (n_embd, head_size) -> (B, T, head_size)
            
            # wei = jnp.einsum('bths,buhs->bhtu', q, k) * (head_size ** -0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
            # wei = jnp.where(tril[:T, :T], wei, -jnp.inf)
            # wei = jax.nn.softmax(wei, axis=-1)
            # # wei = apply_dropout(wei, key=key, is_training=is_training)
            # out = jnp.einsum('bhtu,buhs->bths', wei, v) # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

            # More efficient implementation using dot_product_attention
            #NOTE why is this 20 ms slower?
            out = jax.nn.dot_product_attention(q, k, v,is_causal=True)
            out = out.astype(compute_dtype).reshape(B, T, -1) # (B, T, head_size)
            out = out @ params['W_out'][layer_idx]# (B, T, num_heads * head_size) @ (num_heads * head_size, n_embd) -> (B, T, n_embd)
            out = apply_dropout(out, key=key, is_training=is_training)
            return out
    def transformer_block(x, layer_idx, key=None, is_training=False):
        x = x + multi_head_attention(layer_norm(x, params['ln1_gamma'][layer_idx], params['ln1_beta'][layer_idx]), layer_idx=layer_idx, key=key,is_training=is_training) # (B,T,n_embd)
        x = x + feedforward(layer_norm(x, params['ln2_gamma'][layer_idx], params['ln2_beta'][layer_idx]), layer_idx=layer_idx) # (B,T,n_embd)
        return x
    B,T = idx.shape

    token_embeddings = jnp.take(params['token_embedding'], idx, axis=0).astype(jnp.float32) # (B, T, n_embd)
    positional_embeddings = params['positional_embedding'][jnp.arange(T)].astype(jnp.float32) # (T, n_embd)
    x = token_embeddings + positional_embeddings
    for i in range(n_layer): # Just one block for simplicity
        x = transformer_block(x, layer_idx=i, key=key, is_training=is_training)
    x = layer_norm(x, params['ln_f_gamma'], params['ln_f_beta']) # (B, T, n_embd)
    x = x @ params['token_embedding'].T # (B, T, n_embd) @ (n_embd, vocab_size) -> (B, T, vocab_size)

    return x.astype(jnp.float32) # (B, T, vocab_size) 

@partial(jax.jit, static_argnames=['max_new_tokens', 'temperature', 'top_k'])
def generate(params, prompt_tokens, max_new_tokens=100, temperature=1.0, top_k=None, key=jax.random.PRNGKey(0)):
    """
    prompt_tokens: list or 1D array of token ids
    """
    # Start with shape (1, T)
    idx = jnp.array(prompt_tokens, dtype=jnp.int32)[None, :]

    for _ in range(max_new_tokens):
        # Crop to block_size if too long
        idx_cond = idx[:, -block_size:]
        logits = forward(params, idx_cond, is_training=False, key=key)  # (1, T, vocab_size)
        logits = logits[:, -1, :]  # (1, vocab_size)
        logits = logits / temperature
        # Optional top-k sampling
        if top_k is not None:
            top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
            # Set everything outside top-k to -inf
            logits = jnp.full_like(logits, -jnp.inf)
            logits = logits.at[0, top_k_indices[0]].set(top_k_logits[0])
        key, subkey = jax.random.split(key)
        next_token = jax.random.categorical(subkey, logits, axis=-1)  # (1,)

        # Append to sequence
        idx = jnp.concatenate([idx, next_token[:, None]], axis=1)

    return idx[0]  # return 1D array of tokens

@jax.jit
def loss_fn(params, idx, targets, key=None):
    logits = forward(params, idx, is_training=True, key=key) # (B, T, vocab_size)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    targets = targets.reshape(B*T)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets).mean()
    return loss

@jax.jit
def accume_step(params, idx, targets, key=None):
    loss, grad = jax.value_and_grad(loss_fn)(params, idx, targets, key=key)
    return loss, grad

def eval_model(params, key):
    xb, yb = get_eval_batch(val_data, key)
    val_loss = loss_fn(params, xb, yb, key=key)
    return val_loss
# --- Training Loop ---
params = init_model_params(subkey, vocab_size, n_embd)

print(f"Training on {device}...")
key, train_key = jax.random.split(key)


schedular = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=max_lr,decay_steps=max_iters,end_value=min_lr, warmup_steps=warmup_steps)
optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                        optax.adamw(schedular, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, mask= lambda p: jax.tree.map(lambda x: x.ndim >= 2, p)),
                        )
optimizer = optax.apply_if_finite(optimizer,max_consecutive_errors=5)
opt_state = optimizer.init(params)

devices = jax.devices()
print(f"Available devices: {devices}")
mesh = Mesh(devices, ('data',))
print(f"Using mesh: {mesh}")

num_chunks = 103  # full fineweb10B. Each chunk is 100M tokens
steps_per_chunk = (100_000_000) // (batch_size * block_size * grad_accum_steps)
with mesh:
    params = jax.device_put(params, NamedSharding(mesh, P()))
    opt_state = jax.device_put(opt_state, NamedSharding(mesh, P()))
    for step in range(max_iters):
        if step % steps_per_chunk == 0:
            chunk_idx = (step // steps_per_chunk) % num_chunks + 1
            current_chunck = load_chunk(chunk_idx)
        accume_grad = None
        total_loss = 0.0
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            train_key, subkey = jax.random.split(train_key)
            xb, yb = get_batch(current_chunck, subkey)
            xb, yb = jax.device_put((xb, yb), NamedSharding(mesh, P('data', None)))
            loss, grad = accume_step(params, xb, yb,key=subkey)
            total_loss += loss
            if accume_grad is None:
                accume_grad = grad
            else:
                accume_grad = jax.tree.map(lambda g1, g2: g1 + g2, accume_grad, grad)
        avg_loss = float(total_loss) / grad_accum_steps
        accume_grad = jax.tree.map(lambda g: g / grad_accum_steps, accume_grad)
        norm = optax.global_norm(accume_grad)
        updates, opt_state = optimizer.update(accume_grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss.block_until_ready()  # Ensure loss is computed before timing
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (batch_size * block_size * grad_accum_steps) / (t1 - t0)
        wandb.log({
                "train/loss": avg_loss,
                "train/grad_norm": float(norm),
                "train/tokens_per_sec": tokens_per_sec,
                "train/lr": float(schedular(step)),
            }, step=step)
        if step % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            val_loss = eval_model(params, eval_key)
            print(f"Step {step}: train loss {total_loss / grad_accum_steps}, val loss {val_loss} in {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")


# write params to disk
import pickle
import numpy as np

print("Saving model parameters...")
# 1. Move parameters to CPU
# 2. Convert to standard numpy float32 (NumPy doesn't natively support bfloat16)
params_cpu = jax.tree.map(lambda x: np.array(x, dtype=np.float32), params)

with open("gpt2_params.pkl", "wb") as f:
    pickle.dump(params_cpu, f)
    
print("Model saved successfully as gpt2_params.pkl!")
# # Generate
# output_tokens = generate(
#     params,
#     prompt_tokens=tokens,
#     max_new_tokens=50,
#     temperature=0.8,
#     top_k=40,
#     key=jax.random.PRNGKey(42)
# )

# # Decode
# print(enc.decode(output_tokens.tolist()))