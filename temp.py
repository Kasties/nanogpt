import jax
key = jax.random.PRNGKey(1337)
print(key)
keys = jax.random.split(key=key, num=20) # We need a lot of random keys, so we split the key into many subkeys at once
print(keys[0])