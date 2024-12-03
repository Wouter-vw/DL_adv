import os
import jax.random as jrandom
import jax.numpy as jnp
import numpy as np
import sklearn.model_selection

def load_banana_data(shuffle=True):
    # We have to load the banana dataset
    filen = os.path.join("data", "banana", "banana.csv")
    Xy = np.loadtxt(filen, delimiter=",")
    Xy = jnp.asarray(Xy)
    x_train, y_train = Xy[:, :-1], Xy[:, -1]
    x_test, y_test = Xy[:0, :-1], Xy[:0, -1]
    y_train, y_test = y_train - 1, y_test - 1

    split_train_size = 0.7
    strat = None
    x_full, y_full = jnp.concatenate((x_train, x_test)), jnp.concatenate((y_train, y_test)) # full dataset
    # Split the data into train, valid, test
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x_full, y_full, train_size=split_train_size, random_state=230, shuffle=shuffle, stratify=strat
    )
    x_test, x_valid, y_test, y_valid = sklearn.model_selection.train_test_split(
        x_test, y_test, train_size=0.5, random_state=230, shuffle=shuffle, stratify=strat
    )

    x_train = x_train[:265, :]
    y_train = y_train[:265]

    # define train , valid, test, create data loader for training
    x_train = x_train.astype(jnp.float64)
    x_valid = x_valid.astype(jnp.float64)
    x_test = x_test.astype(jnp.float64)
    # labels are integers
    y_train = y_train.astype(jnp.int64)
    y_valid = y_valid.astype(jnp.int64)
    y_test = y_test.astype(jnp.int64)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

# Review if there is a more efficient method for this!
# Shuffle and batch manually
def create_data_loader(x_data, y_data, batch_size, rng, shuffle=True):
    dataset_size = len(x_data)
    indices = jnp.arange(dataset_size)
    
    if shuffle:
        indices = jrandom.permutation(rng, dataset_size, independent=True)
    
    # Create batches
    batches = []
    for i in range(0, dataset_size, batch_size):
        batch_idx = indices[i:i+batch_size]
        batches.append((x_data[batch_idx], y_data[batch_idx]))
    
    return batches
