import jax
import jax.numpy as jnp
import optax
from functools import partial

# --- Hyperparameters ---
batch_size = 32
block_size = 8
max_iters = 30000
learning_rate = 1e-2
device = 'tpu' # JAX handles this automatically usually
vocab_size = 65 # Default for shakespeare char-level
eval_interval = 300
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
def init_model_params(key, vocab_size):
    return jax.random.normal(key, (vocab_size, vocab_size)) * 0.01

def forward(params, idx):
    logits = jnp.take(params, idx, axis=0)
    return logits

@jax.jit
def loss_fn(params, idx, targets):
    logits = forward(params, idx)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    targets = targets.reshape(B*T)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss

# --- Fix applied here: static_argnames ---
@partial(jax.jit, static_argnames=['max_new_tokens'])
def generate(params, idx, max_new_tokens, key):
    current_idx = idx[:, -1] # (B,)
    
    def step_fn(carry, _):
        curr_idx, rng = carry
        logits = jnp.take(params, curr_idx, axis=0)
        rng, subkey = jax.random.split(rng)
        # Sample
        next_token = jax.random.categorical(subkey, logits, axis=-1)
        return (next_token, rng), next_token

    _, generated_tokens = jax.lax.scan(step_fn, (current_idx, key), length=max_new_tokens)
    
    # scan outputs (Time, Batch), we want (Batch, Time)
    generated_tokens = generated_tokens.T 
    return jnp.concatenate((idx, generated_tokens), axis=1)

# --- Training Setup ---
rng = jax.random.PRNGKey(1337)
key, subkey = jax.random.split(rng)
params = init_model_params(subkey, vocab_size)

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

# 1. Start with a zero token (Batch size 1, Sequence length 1)
start_idx = jnp.zeros((1, 1), dtype=jnp.int32) 

# 2. Compile and run generation
# max_new_tokens is now static, so JAX compiles this specifically for length 300
generated = generate(params, start_idx, max_new_tokens=300, key=gen_key)

# 3. Decode
print(decode(generated[0].tolist()))