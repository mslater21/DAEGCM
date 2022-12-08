import torch
import utility
from pretrain import pretrain_model
from train import train_model
from scoring import get_average_scores
from IPython.display import clear_output


## Set desired settings here ###
dataset_name = 'Cora'  # Cora, CiteSeer, or PubMed - PubMed is very resource heavy, may crash runtime

if dataset_name == 'Cora':
  num_classes = 7
elif dataset_name == 'CiteSeer':
  num_classes = 6
elif dataset_name == 'PubMed':
  num_classes = 3

models_to_train = 50  # Number of models to train, average scores will be reported.
verbose = False  # Set to true to turn off clearing epoch scores after each model is done training

# Pretrain model settings
max_pretrain_epochs = 30
pretrain_lr = .005
pretrain_weight_decay = 5e-3

# Post-pretrain model settings (encoder + clustering)
max_training_epochs = 100
update_interval = 5
training_lr = .001
training_weight_decay = 5e-3
clustering_bias = 10
objective_multiplier = .5  # .5 is default, 0 = no clustering loss, 1 = no encoder loss
similarity = 'euclidean'
bin_method = 'equal_count'

## End of settings section ##

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X, y, adj, adj_label, M = utility.prep_dataset(dataset_name, device)

best_scores_array = []
for i in range(models_to_train):
  print('Training model', i)
  pretrained, history = pretrain_model(X, y, adj, adj_label, M, num_classes, max_pretrain_epochs, device, pretrain_lr, pretrain_weight_decay)

  model, history, best_scores = train_model(X, y, adj, adj_label, M, pretrained, history, num_classes, max_training_epochs, device, update_interval, training_lr, training_weight_decay, clustering_bias, objective_multiplier, similarity, bin_method)

  best_scores_array.append(best_scores)
  if verbose:
    clear_output()

  get_average_scores(best_scores_array)