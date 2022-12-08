import torch
import torch.nn as nn
import torch.nn.functional as F

from galayer import GALayer


class GAEncoder(nn.Module):
    def __init__(self, num_features, hidden_size=256, embedding_size=16, alpha=.2):
        super(GAEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GALayer(num_features, hidden_size, alpha)
        self.conv2 = GALayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred