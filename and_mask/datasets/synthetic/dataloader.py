import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import make_classification, make_blobs
from and_mask.datasets.synthetic.synthetic_data_gen import get_spirals_dataset, get_strong_weak_dataset

def make_dataloader(n_examples, env, n_envs, n_revolutions, n_dims,
                    batch_size,
                    use_cuda,
                    flip_first_signature=False,
                    seed=None):


    inputs, labels = get_strong_weak_dataset(n_samples=n_examples,glob_sigma=1, sig_sigma=1, K=n_dims+2, rand_strong_coefs=3., inv_weak_coefs=0.3,\
                                                 n_informative=2, n_redundant=0, use_sklearn=False, batch_size=batch_size, seed=seed)

    if flip_first_signature:
        inputs[:1, 2:] = -inputs[:1, 2:]

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    data_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.tensor(inputs), torch.tensor(labels)),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    return data_loader

