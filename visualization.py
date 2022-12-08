import torch
import matplotlib.pyplot as plt


def plot_accuracy(history):
    plt.plot(range(len(history['accuracy'])), history['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


def plot_loss(history, training=True, embedding=True, cluster=True):
    labels = []
    if training:
        plt.plot(len(history['training_loss']), history['training_loss'])
        labels.append('Training Loss')
    if embedding:
        plt.plot(len(history['embedding_loss']), history['embedding_loss'])
        labels.append('Embedding Loss')
    if cluster:
        plt.plot(len(history['cluster_loss']), history['cluster_loss'])
        labels.append('Clustering Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Embedding Loss', 'Clustering Loss'])


def plot_histogram(model, X, adj, M, device, bin_method='equal_count'):
    A_pred, z, Q = model(X, adj, M)
    combined = torch.cat((z.clone().detach(), model.cluster_layer.clone().detach()))
    histograms = []
    bins = torch.zeros((len(combined[0]), 101)).to(device)
    counts = torch.zeros((len(combined[0]), 100)).to(device)
    for i in range(8, 9):
        first = combined[:, i].cpu()
        if bin_method == 'equal_count':
            histogram = torch.histogram(first, torch.tensor(model.equalObs(first, 100)).float())
        else:
            histogram = torch.histogram(first, 100)
        bins[i] = histogram[1]
        counts[i] = histogram[0]
        histograms.append([histogram[0].to(device), histogram[1].to(device)])
        plt.hist(histogram[1].data.numpy()[:-1], histogram[1].data.numpy(), weights=histogram[0].data.numpy(),
                 edgecolor='black')
        plt.xlabel('Embedding Value')
        plt.ylabel('Count')
