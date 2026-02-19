import jax
import jax.numpy as jnp
import optax
from functools import partial

# --- Hyperparameters ---
batch_size = 16
block_size = 1024
max_iters = 10000
learning_rate = 3e-4
device = 'tpu' 
vocab_size = 50304 
eval_interval = 300
n_embd = 768
n_layer = 12
dropout = 0.2
num_heads = 12
dtype = jnp.bfloat16
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 1000
tril = jnp.tril(jnp.ones((block_size, block_size), dtype=bool))

rng = jax.random.PRNGKey(1337)
key, subkey = jax.random.split(rng)

import tiktoken

enc = tiktoken.get_encoding("gpt2")
with open("input.txt", "r") as f:
    data = f.read()
tokens = enc.encode(data)
data_arr = jnp.array(tokens, dtype=jnp.int32)
n = int(0.9*len(data_arr))
train_data = data_arr[:n]
val_data = data_arr[n:]


@partial(jax.jit, static_argnames=['split'])
def get_batch(split, key):
    data_arr = train_data if split == 'train' else val_data
    
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


def init_model_params(key, vocab_size, n_embd):
    
    head_size = n_embd // num_heads
    params = {
        'token_embedding': jax.random.normal(key, (vocab_size, n_embd),dtype=dtype) * 0.02,
        'positional_embedding': jax.random.normal(key, (block_size, n_embd),dtype=dtype) * 0.01,
        'W_q': [jax.random.normal(key, (num_heads, n_embd, head_size),dtype=dtype, ) * 0.02 for _ in range(n_layer)],
        'W_k': [jax.random.normal(key, (num_heads, n_embd, head_size),dtype=dtype) * 0.02 for _ in range(n_layer)],
        'W_v': [jax.random.normal(key, (num_heads, n_embd, head_size),dtype=dtype) * 0.02 for _ in range(n_layer)],
        'W_out': [jax.random.normal(key, (num_heads * head_size, n_embd),dtype=dtype) * (0.02 / jnp.sqrt(2*n_layer)) for _ in range(n_layer)],
        'W_ffwd': [jax.random.normal(key, (n_embd, 4*n_embd),dtype=dtype) * 0.02 for _ in range(n_layer)],
        'W_ffwd_project': [jax.random.normal(key, (4*n_embd, n_embd),dtype=dtype) * (0.02 / jnp.sqrt(2*n_layer)) for _ in range(n_layer)],
        'ln1_gamma': [jnp.ones((n_embd,),dtype=dtype) for _ in range(n_layer)],
        'ln1_beta':  [jnp.zeros((n_embd,),dtype=dtype) for _ in range(n_layer)],
        'ln2_gamma': [jnp.ones((n_embd,),dtype=dtype) for _ in range(n_layer)],
        'ln2_beta':  [jnp.zeros((n_embd,),dtype=dtype) for _ in range(n_layer)],
        'ln_f_gamma': jnp.ones((n_embd,),dtype=dtype),
        'ln_f_beta':  jnp.zeros((n_embd,),dtype=dtype),
    }
    return params

@partial(jax.jit, static_argnames=['is_training'])
def forward(params, idx, is_training=False, target=None, key=None):
    def layer_norm(x,gamma,beta,eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / jnp.sqrt(var + eps) + beta
    def apply_dropout(x, rate=dropout, key=None, is_training=False):
        if not is_training or key is None:
            return x
        keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
        return x * keep / (1.0 - rate)

    def feedforward(x, layer_idx):
        out = jax.nn.gelu(x @ params['W_ffwd'][layer_idx],approximate=True) # (B, T, n_embd) @ (n_embd, 4*n_embd) -> (B, T, 4*n _embd) use aproximate gelu as it was the original activation in GPT2
        out = out @ params['W_ffwd_project'][layer_idx] # (B, T, 4*n_embd) @ (4*n_embd, n_embd) -> (B, T, n_embd)
        # out = apply_dropout(out, key=jax.random.PRNGKey(42), is_training=is_training) # No dropout for simplicity
        return out

    def multi_head_attention(x, layer_idx, key=None, is_training=False):
            # (B, T, n_embd) -> (B, T, num_heads * head_size)
            q = jnp.einsum('bte,hes->bths', x, params['W_q'][layer_idx]) # (B, T, n_embd) @ (n_embd, head_size) -> (B, T, head_size)
            k = jnp.einsum('bte,hes->bths', x, params['W_k'][layer_idx]) # (B, T, n_embd) @ (n_embd, head_size) -> (B, T, head_size)
            v = jnp.einsum('bte,hes->bths', x, params['W_v'][layer_idx]) # (B, T, n_embd) @ (n_embd, head_size) -> (B, T, head_size)
            
            wei = jnp.einsum('bths,buhs->bhtu', q, k) * (head_size ** -0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
            wei = jnp.where(tril[:T, :T], wei, -jnp.inf)
            wei = jax.nn.softmax(wei, axis=-1)
            # wei = apply_dropout(wei, key=key, is_training=is_training)
            out = jnp.einsum('bhtu,buhs->bths', wei, v) # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

            # More efficient implementation using dot_product_attention
            #NOTE why is this 20 ms slower?
            # out = jax.nn.dot_product_attention(q, k, v,is_causal=True).reshape(B, T, -1) # (B, T, head_size)
            out = out.reshape(B, T, -1) # (B, T, num_heads * head_size)
            out = out @ params['W_out'][layer_idx] # (B, T, num_heads * head_size) @ (num_heads * head_size, n_embd) -> (B, T, n_embd)
            out = apply_dropout(out, key=key, is_training=is_training)
            return out
    def transformer_block(x, layer_idx, key=None, is_training=False):
        x = x + multi_head_attention(layer_norm(x, params['ln1_gamma'][layer_idx], params['ln1_beta'][layer_idx]), layer_idx=layer_idx, key=key,is_training=is_training) # (B,T,n_embd)
        x = x + feedforward(layer_norm(x, params['ln2_gamma'][layer_idx], params['ln2_beta'][layer_idx]), layer_idx=layer_idx) # (B,T,n_embd)
        return x
    B,T = idx.shape

    token_embeddings = jnp.take(params['token_embedding'], idx, axis=0) # (B, T, n_embd)
    positional_embeddings = params['positional_embedding'][jnp.arange(T)] # (T, n_embd)
    head_size = n_embd // num_heads
    x = token_embeddings + positional_embeddings
    for i in range(n_layer): # Just one block for simplicity
        x = transformer_block(x, layer_idx=i, key=key, is_training=is_training)
    x = layer_norm(x, params['ln_f_gamma'], params['ln_f_beta']) # (B, T, n_embd)
    x = x @ params['token_embedding'].T # (B, T, n_embd) @ (n_embd, vocab_size) -> (B, T, vocab_size)

    return x # (B, T, vocab_size) 

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
        logits = forward(params, idx_cond, is_training=False)
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
def loss_fn(params, idx, targets):
    logits = forward(params, idx, is_training=True) # (B, T, vocab_size)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    targets = targets.reshape(B*T)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss

@jax.jit
def training_step(params, opt_state, idx, targets):
    loss, grad = jax.value_and_grad(loss_fn)(params, idx, targets)
    norm = optax.global_norm(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss ,norm

def eval_model(params, key):
    xb, yb = get_batch('val', key)
    val_loss = loss_fn(params, xb, yb)
    return val_loss
# --- Training Loop ---
params = init_model_params(subkey, vocab_size, n_embd)

print(f"Training on {device}...")
key, train_key = jax.random.split(key)


schedular = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=max_lr,decay_steps=max_iters,end_value=min_lr, warmup_steps=warmup_steps)
optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                        optax.adamw(schedular, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, mask= lambda p: jax.tree.map(lambda x: x.ndim >= 2, p)),
                        )
opt_state = optimizer.init(params)

import time
for step in range(max_iters):
    train_key, subkey = jax.random.split(train_key)
    xb, yb = get_batch('train', subkey)
    t0 = time.time()
    params, opt_state, loss, norm = training_step(params, opt_state, xb, yb)
    loss.block_until_ready()  # Ensure loss is computed before timing
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (batch_size * block_size) / (t1 - t0)
    print(f"Step {step}: train loss {loss}, norm {norm} in {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")
    if step % eval_interval == 0:
        key, eval_key = jax.random.split(key)
        val_loss = eval_model(params, eval_key)
        print(f"Step {step}: train loss {loss}, val loss {val_loss} in {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")




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