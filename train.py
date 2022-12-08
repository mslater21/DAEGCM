import torch
import torch.nn.functional as F
from daegc import DAEGC
from torch.optim import Adam
from sklearn.cluster import KMeans
from scoring import score
import numpy as np
from utility import target_distribution


def train_model(X, y, adj, adj_label, M, pretrained_model, history, num_classes, epochs, device, update_interval=5,
                lr=.0001, weight_decay=5e-3, clustering_bias=10, objective_multiplier=None, similarity='euclidean',
                bin_method='equal_width'):
    model = DAEGC(pretrained_model, len(X[0]), num_clusters=num_classes, similarity=similarity,
                  bin_method=bin_method).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print('Objective Multiplier: ', objective_multiplier)
    best_scores = (0, [])

    # Get initial node embeddings
    with torch.no_grad():
        _, z = model.gat(X, adj, M)

        # Get initial clustering centroids
        kmeans = KMeans(num_classes, n_init=20)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
        score(y, y_pred, 'init')

    for epoch in range(epochs):
        model.train()
        A_pred, z, Q = model(X, adj, M)
        if epoch % update_interval == 0:
            q = Q.detach().data.cpu().numpy().argmax(1)
            scores = score(y, q, epoch)
            history['accuracy'].append(scores[0])
            history['NMI'].append(scores[1])
            history['ARI'].append(scores[2])
            history['F1'].append(scores[3])
            if np.sum(scores) > best_scores[0]:
                best_scores = (np.sum(scores), scores)
            p = target_distribution(Q.detach())

        kl_loss = F.kl_div(Q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        if objective_multiplier is None:
            loss = clustering_bias * kl_loss + re_loss
        else:
            loss = objective_multiplier * clustering_bias * kl_loss + (1 - objective_multiplier) * re_loss

        history['training_loss'].append(loss.item())
        history['cluster_loss'].append(kl_loss.item())
        history['embedding_loss'].append(re_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, history, best_scores
