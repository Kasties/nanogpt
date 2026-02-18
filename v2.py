import jax
import jax.numpy as jnp
import optax
from functools import partial

# --- Hyperparameters ---
batch_size = 64
block_size = 256
max_iters = 10000
learning_rate = 3e-4
device = 'tpu' # JAX handles this automatically usually
vocab_size = 65 # Default for shakespeare char-level
eval_interval = 300
n_embd = 384
n_layer = 6
dropout = 0.2
# --- Data Loading ---
# Creating dummy data if input.txt is missing for this example
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("input.txt not found, using dummy data.")
    text = "Here is some dummy text to make the model run. " * 1000

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# --- Optimized Batch Fetching ---
# Use vmap + dynamic_slice instead of list comprehensions for TPU speed
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

# --- Model Definition ---
def init_model_params(key, vocab_size, n_embd):
    num_heads = 4
    head_size = n_embd // num_heads
    layer_keys = jax.random.split(key, n_layer * 10) # 10 matrices per layer
    layer = {'token_embedding': jax.random.normal(layer_keys[0], (vocab_size, n_embd)) * (n_embd ** -0.5),
            'positional_embedding': jax.random.normal(layer_keys[1], (block_size, n_embd)) * 0.01,
            'W_k': jax.random.normal(layer_keys[2], (n_layer,num_heads, n_embd, head_size)) * (n_embd ** -0.5),
            'W_q': jax.random.normal(layer_keys[3], (n_layer,num_heads, n_embd, head_size)) * (n_embd ** -0.5),
            'W_v': jax.random.normal(layer_keys[4], (n_layer,num_heads, n_embd, head_size)) * (n_embd ** -0.5),
            'W_out': jax.random.normal(layer_keys[5], (n_layer,num_heads * head_size, n_embd)) * (n_embd ** -0.5),
            'W_ffwd': jax.random.normal(layer_keys[6], (n_layer,n_embd,4 * n_embd)) * (n_embd ** -0.5),
            'W_lm_head': jax.random.normal(layer_keys[7], (n_embd, vocab_size)) * (n_embd ** -0.5),
            'W_project': jax.random.normal(layer_keys[8], (n_layer,n_embd, n_embd)) * (n_embd ** -0.5),
            'W_ffwd_project': jax.random.normal(layer_keys[9], (n_layer,4 * n_embd, n_embd)) * (n_embd ** -0.5)
            }
    return layer


def forward(params, idx, is_training=False, target=None):
    def layer_norm(x, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + eps)
    def apply_dropout(x, rate=dropout, key=None, is_training=False):
        if not is_training or key is None:
            return x
        keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
        return x * keep / (1.0 - rate)
    def single_head(W_k, W_q, W_v,x):
        k = x @ W_k # (B, T, head_size)
        q = x @ W_q # (B, T, head_size)
        wei = q @ k.transpose(0,2,1) * (head_size ** -0.5) # (B, T, 16) @ (B, head_size, T) -> (B, T, T) 
        wei = jnp.where(jnp.tril(jnp.ones((T, T), dtype=bool)), wei, -jnp.inf)
        wei = jax.nn.softmax(wei, axis=-1)
        wei = apply_dropout(wei, key=jax.random.PRNGKey(42), is_training=is_training) 
        v = x @ W_v
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

    def feedforward(x, layer_idx):
        out = jax.nn.relu(x @ params['W_ffwd'][layer_idx]) # (B, T, n_embd) @ (n_embd, 4*n_embd) -> (B, T, 4*n_embd)
        out = out @ params['W_ffwd_project'][layer_idx] # (B, T, 4*n_embd) @ (4*n_embd, n_embd) -> (B, T, n_embd)
        out = apply_dropout(out, key=jax.random.PRNGKey(42), is_training=is_training) # No dropout for simplicity
        return out
    
    def multi_head_attention(x, layer_idx):
            # (B, T, n_embd) -> (B, T, num_heads * head_size)
            heads_out = jax.vmap(single_head, in_axes=(0, 0, 0, None))(params['W_k'][layer_idx], params['W_q'][layer_idx], params['W_v'][layer_idx], x) # (num_heads, B, T, head_size)
            heads_out = heads_out.transpose(1, 2, 0, 3).reshape(B, T, -1) # (B, T, num_heads * head_size)
            out = heads_out @ params['W_out'][layer_idx] # (B, T, num_heads * head_size) @ (num_heads * head_size, vocab_size) -> (B, T, vocab_size)
            out = out @ params['W_project'][layer_idx] # (B, T, n_embd) @ (n_embd, n_embd) -> (B, T, n_embd)¨
            out = apply_dropout(out, key=jax.random.PRNGKey(42), is_training=is_training)
            return out
    def transformer_block(x, layer_idx):
        x = x + multi_head_attention(layer_norm(x), layer_idx=layer_idx) # (B, T, n_embd)
        x = x + feedforward(layer_norm(x), layer_idx=layer_idx) # (B, T, n_embd)
        return x
    B,T = idx.shape

    token_embeddings = jnp.take(params['token_embedding'], idx, axis=0) # (B, T, n_embd)
    positional_embeddings = params['positional_embedding'][jnp.arange(T)] # (T, n_embd)
    head_size = n_embd // 4
    x = token_embeddings + positional_embeddings
    for i in range(n_layer): # Just one block for simplicity
        x = transformer_block(x, layer_idx=i)
    x = layer_norm(x) # (B, T, n_embd)
    x = x @ params['W_lm_head'] # (B, T, n_embd) @ (n_embd, vocab_size) -> (B, T, vocab_size)

    return x # (B, T, vocab_size)


@jax.jit
def loss_fn(params, idx, targets):
    logits = forward(params, idx, target=targets, is_training=True) # (B, T, vocab_size)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    targets = targets.reshape(B*T)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss

# --- Fix applied here: static_argnames ---
@partial(jax.jit, static_argnames=['max_new_tokens'])
def generate(params, idx, max_new_tokens, key):
    B, T0 = idx.shape
    buf = jnp.concatenate([idx, jnp.zeros((B, max_new_tokens), dtype=jnp.int32)], axis=1)

    def step_fn(carry, t):
        buf, rng = carry
        ctx = jax.lax.dynamic_slice(buf, (0, t), (B, block_size))
        logits = forward(params, ctx, is_training=False)[:, -1, :]  # (B, vocab_size)
        rng, subkey = jax.random.split(rng)
        next_token = jax.random.categorical(subkey, logits, axis=-1)  # (B,)
        buf = buf.at[:, t + block_size].set(next_token)
        return (buf, rng), None

    (buf, _), _ = jax.lax.scan(step_fn, (buf, key), jnp.arange(max_new_tokens))
    return buf

# --- Training Setup ---
rng = jax.random.PRNGKey(1337)
key, subkey = jax.random.split(rng)
params = init_model_params(subkey, vocab_size, n_embd)

optimizer = optax.adamw(learning_rate)
opt_state = optimizer.init(params)

@jax.jit
def training_step(params, opt_state, idx, targets):
    loss, grad = jax.value_and_grad(loss_fn)(params, idx, targets)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def eval_model(params, key):
    xb, yb = get_batch('val', key)
    val_loss = loss_fn(params, xb, yb)
    return val_loss
# --- Training Loop ---
print(f"Training on {device}...")
key, train_key = jax.random.split(key)

for step in range(max_iters):
    train_key, subkey = jax.random.split(train_key)
    xb, yb = get_batch('train', subkey)
    
    params, opt_state, loss = training_step(params, opt_state, xb, yb)
    
    if step % eval_interval == 0:
        key, eval_key = jax.random.split(key)
        val_loss = eval_model(params, eval_key)
        print(f"Step {step}: train loss {loss:.4f}, val loss {val_loss:.4f}")


print(f"Final Loss: {loss:.4f}")

# --- Generation ---
print("Generating...")
key, gen_key = jax.random.split(key)

start_idx = jnp.zeros((1, 1), dtype=jnp.int32) 

generated = generate(params, start_idx, max_new_tokens=300, key=gen_key)

print(decode(generated[0].tolist()))