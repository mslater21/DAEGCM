import torch
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import normalize
import numpy as np


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def get_dataset(dataset_name):
    dataset = Planetoid('./dataset', dataset_name)
    return dataset


def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]),
        torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset


def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    t = 2  # May need to tweak
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


def prep_dataset(dataset_name, device):
    dataset = get_dataset(dataset_name)[0]
    dataset = data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = get_M(adj).to(device)

    X = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()
    return X, y, adj, adj_label, M