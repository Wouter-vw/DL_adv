import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax


# Define your model using Flax
class MLP(nn.Module):
    num_features: int
    hidden_size: int
    num_output: int

    @nn.compact
    def __call__(self, x):
        self.sow('intermediates', 'in_0', x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        self.sow('intermediates', 'in_1', x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        self.sow('intermediates', 'in_2', x)
        x = nn.Dense(self.num_output)(x)
        return x


# Create training state
def create_train_state(rng, model, optimizer):
    params = model.init(rng, jnp.ones([1, model.num_features]))  # Dummy input for parameter initialization
    if optimizer == "sgd":
        learning_rate = 1e-3
        weight_decay = 1e-2
        optimizer = optax.sgd(learning_rate)
    else:
        learning_rate = 1e-3
        weight_decay = 1e-3
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)

    apply_fn = jax.jit(model.apply)

    return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer)

    # Loss function
@jax.jit
def compute_loss(logits, labels):
    labels = jnp.asarray(labels, dtype=jnp.int32)  # Ensure labels are int32
    return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels))

# Training function
@jax.jit
def train_step(state, batch_img, batch_label):
    def loss_fn(params):
        logits = state.apply_fn(params, batch_img)
        return compute_loss(logits, batch_label)

    # Compute gradients
    grad = jax.grad(loss_fn)(state.params)
    # Update the state with the new parameters
    new_state = state.apply_gradients(grads=grad)
    return new_state