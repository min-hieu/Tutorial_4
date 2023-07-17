import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets

def normalize(ds, scaling_factor=2.):
    return (ds - ds.mean()) / \
            ds.std() * scaling_factor


def sample_checkerboard(n):
    # https://github.com/ghliu/SB-FBSDE/blob/main/data.py
    n_points = 3*n
    n_classes = 2
    freq = 5
    x = np.random.uniform(-(freq//2)*np.pi, (freq//2)*np.pi, size=(n_points, n_classes))
    mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
           np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0))
    y = np.eye(n_classes)[1*mask]
    x0 = x[:,0]*y[:,0]
    x1 = x[:,1]*y[:,0]
    sample = np.concatenate([x0[...,None],x1[...,None]],axis=-1)
    sqr    = np.sum(np.square(sample),axis=-1)
    idxs   = np.where(sqr==0)
    sample = np.delete(sample,idxs,axis=0)

    return sample


def load_twodim(num_samples: int,
                dataset: str,
                dimension: int = 2):

    if dataset == 'gaussian_centered':
        sample = np.random.normal(size=(num_samples, dimension))
        sample = sample

    if dataset == 'gaussian_shift':
        sample = np.random.normal(size=(num_samples, dimension))
        sample = sample + 1.5

    if dataset == 'circle':
        X, y = datasets.make_circles(
            n_samples=num_samples, noise=0.0, random_state=None, factor=.5)
        sample = X * 4

    if dataset == 'scurve':
        X, y = datasets.make_s_curve(
            n_samples=num_samples, noise=0.0, random_state=None)
        sample = normalize(X[:, [0, 2]])

    if dataset == 'moon':
        X, y = datasets.make_moons(
            n_samples=num_samples, noise=0.0, random_state=None)
        sample = normalize(X)

    if dataset == 'swiss_roll':
        X, y = datasets.make_swiss_roll(
            n_samples=num_samples, noise=0.0, random_state=None, hole=True)
        sample = normalize(X[:, [0, 2]])

    if dataset == 'checkerboard':
        sample = normalize(sample_checkerboard(num_samples))

    return torch.tensor(sample).float()


class TwoDimDataClass(Dataset):

    def __init__(self,
                 dataset_type: str,
                 N: int,
                 batch_size: int,
                 dimension = 2):

        self.X = load_twodim(N, dataset_type, dimension=dimension)
        self.name = dataset_type
        self.batch_size = batch_size
        self.dimension = 2

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

    def get_dataloader(self, shuffle=True):
        return DataLoader(self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
