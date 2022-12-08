import torch
import torch.nn.functional as F
from gaencoder import GAEncoder
from torch.optim import Adam
from sklearn.cluster import KMeans
from copy import deepcopy
from scoring import score
import numpy as np


def pretrain_model(X, y, adj, adj_label, M, num_classes, epochs, device, lr=0.05, weight_decay=5e-3):
    history = {'training_loss': [], 'embedding_loss': [], 'cluster_loss': [], 'accuracy': [], 'NMI': [], 'ARI': [], 'F1': []}

    pretrain_model = GAEncoder(len(X[0])).to(device)
    optimizer = Adam(pretrain_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_scores = (0, 0)
    best_model = None

    for epoch in range(epochs):
        pretrain_model.train()
        A_pred, z = pretrain_model(X, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        history['training_loss'].append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = pretrain_model(X, adj, M)
            kmeans = KMeans(n_clusters=num_classes, n_init=20).fit(z.data.cpu().numpy())
            acc, nmi, ari, f1 = score(y, kmeans.labels_, epoch)
            history['accuracy'].append(acc)
            history['NMI'].append(nmi)
            history['ARI'].append(ari)
            history['F1'].append(f1)
        if np.sum([acc, nmi, ari, f1]) / 4 > best_scores[1]:
            best_scores = (epoch, np.sum([acc, nmi, ari, f1]) / 4)
            best_model = deepcopy(pretrain_model.state_dict())
    return best_model, history