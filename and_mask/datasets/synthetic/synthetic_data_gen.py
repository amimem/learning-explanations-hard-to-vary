import numpy as np
from numpy.random import default_rng

def get_spirals_dataset(n_examples, n_rotations, env, n_envs,
                        n_dims_signatures,
                        seed=None):
    """
    env must either be "test" or an int between 0 and n_envs-1
    n_dims_signatures: how many dimensions for the signatures (spirals are always 2)
    seed: seed for numpy
    """
    assert env == 'test' or 0 <= int(env) < n_envs

    # Generate fixed dictionary of signatures
    rng = np.random.RandomState(seed)

    signatures_matrix = rng.randn(n_envs, n_dims_signatures)

    radii = rng.uniform(0.08, 1, n_examples)
    angles = 2 * n_rotations * np.pi * radii

    labels = rng.randint(0, 2, n_examples)
    angles = angles + np.pi * labels

    radii += rng.uniform(-0.02, 0.02, n_examples)
    xs = np.cos(angles) * radii
    ys = np.sin(angles) * radii

    if env == 'test':
        signatures = rng.randn(n_examples, n_dims_signatures)
    else:
        env = int(env)
        signatures_labels = np.array(labels * 2 - 1).reshape(1, -1)
        signatures = signatures_matrix[env] * signatures_labels.T

    signatures = np.stack(signatures)
    mechanisms = np.stack((xs, ys), axis=1)
    mechanisms /= mechanisms.std(axis=0)  # make approx unit variance (signatures already are)
    inputs = np.hstack((mechanisms, signatures))

    return inputs.astype(np.float32), labels.astype(np.float32)

def get_strong_weak_dataset(n_samples,glob_sigma,sig_sigma,K,rand_strong_coefs,inv_weak_coefs,n_informative,n_redundant,use_sklearn, batch_size, seed):
    # more info for sklearn dataset generation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
    rng = default_rng(seed)
    X = rng.normal(0, glob_sigma, (n_samples, K))
    y = rng.choice(2, size=n_samples, replace=True)

    # pick a subset of best predictors at random 
    true_rng = default_rng(0)
    true_pred_indices = true_rng.choice(K, size=1, replace=False)
    rand_best_indices = rng.choice(K, size=1, replace=False)

    # check to make sure these randoms aren't out true pred
    condition = True
    while condition:
        for i,rbp in enumerate(rand_best_indices):
            if rbp in true_pred_indices:
                #rand_best_indices[i] = np.random.randint(K, size=1)
                rand_best_indices[i] = rng.choice(K, size=1, replace=False)
                condition = True
                break
            else:
                condition = False
    print(true_pred_indices,rand_best_indices)
    # set the coefficients at the right row 
    # that separates each class
    for n in range(n_samples):
        if y[n] == 1:
            X[n,true_pred_indices] = inv_weak_coefs # +  np.random.normal(0,sig_sigma)
            X[n,rand_best_indices] = rand_strong_coefs + rng.normal(0,sig_sigma)# + dist.sample(X[e,rand_best_indices].shape)  #

        elif y[n] == 0:
            X[n,true_pred_indices] = 0 #+  np.random.normal(0,sig_sigma) #-inv_weak_coefs 
            X[n,rand_best_indices] = 0 + rng.normal(0,sig_sigma) # dist.sample(X[e,rand_best_indices].shape)

    return X, y