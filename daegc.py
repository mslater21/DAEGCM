import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from gaencoder import GAEncoder


class DAEGC(nn.Module):
    def __init__(self, pretrained_model, num_features, hidden_size=256, embedding_size=16, alpha=.2, num_clusters=6,
                 similarity='euclidean', bin_method='equal_width'):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.similarity = similarity
        self.bin_method = bin_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gat = GAEncoder(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(pretrained_model)

        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z, self.similarity)

        return A_pred, z, q

    def get_Q(self, z, method='euclidean'):
        if method == 'euclidean':
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))
            q = (q.t() / torch.sum(q, 1)).t()
        elif method == 'manhattan':
            q = 1.0 / (1.0 + torch.sum(torch.abs(torch.pow(z.unsqueeze(1) - self.cluster_layer, 1)), 2))
            q = (q.t() / torch.sum(q, 1)).t()
        elif method == 'cosine':
            q = 1.0 / (1.0 + torch.nn.CosineSimilarity(dim=2)(z.unsqueeze(1), self.cluster_layer))
            q = (q.t() / torch.sum(q, 1)).t()
        elif method == 'mass':
            q = 1.0 / (1.0 + self.mass_dissimilarity(z))
            q = (q.t() / torch.sum(q, 1)).t()
        return q

    def mass_dissimilarity(self, z):
        S, C, F, B = range(4)
        combined = torch.cat((z.clone().detach(), self.cluster_layer.clone().detach()))
        bins = torch.zeros((len(combined[0]), 101)).to(self.device)
        counts = torch.zeros((len(combined[0]), 100)).to(self.device)
        for i in range(len(combined[0])):
            first = combined[:, i].cpu()
            if self.bin_method == 'equal_width':
                histogram = torch.histogram(first, 100)
            else:
                histogram = torch.histogram(first, torch.tensor(self.equalObs(first, 100)).float())
            bins[i] = histogram[1].to(self.device)
            counts[i] = histogram[0].to(self.device)

        sample_bins = torch.searchsorted(bins, z.t()).data - 1
        sample_bins = sample_bins.t()
        cluster_bins = torch.searchsorted(bins, self.cluster_layer.t()) - 1
        cluster_bins = cluster_bins.t()
        max = torch.tensor(
            np.array([torch.maximum(sample_bins[i], cluster_bins).cpu().numpy() for i in range(len(sample_bins))])).to(
            self.device)
        min = torch.tensor(
            np.array([torch.minimum(sample_bins[i], cluster_bins).cpu().numpy() for i in range(len(sample_bins))])).to(
            self.device)
        min = min.unsqueeze(B)
        max = max.unsqueeze(B)
        counts = counts.reshape(1, 1, *counts.shape)
        ns, nc, nf, nb = min.size(S), min.size(C), min.size(F), counts.size(B)
        divisor = len(z) + len(self.cluster_layer)

        cum_counts = counts.cumsum(dim=B).expand(ns, nc, nf, nb).to(self.device)

        is_zero = min <= 0
        lo = (min - 1).masked_fill(is_zero, 0).to(self.device)

        lo_sum = cum_counts.gather(dim=B, index=lo).to(self.device)
        hi_sum = cum_counts.gather(dim=B, index=max).to(self.device)
        sum_counts = torch.where(is_zero, hi_sum, hi_sum - lo_sum)

        pre_sum_output = torch.pow(sum_counts.squeeze(B) / divisor, 2)
        data_mass = torch.sum(pre_sum_output, 2)
        data_mass = torch.pow(data_mass / len(counts), 1 / 2)
        return data_mass

    def equalObs(self, x, nbin):
        nlen = len(x)
        return np.interp(np.linspace(0, nlen, nbin + 1),
                         np.arange(nlen),
                         np.sort(x))