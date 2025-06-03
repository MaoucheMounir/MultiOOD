import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
from torch.utils.data import  TensorDataset, DataLoader

from eval_functions import auc_and_fpr_recall, calc_outputs_test, train, visualize

seed = 42  # Choisis n’importe quel nombre
torch.manual_seed(seed)
np.random.seed(seed)

# Pour les GPU (si applicable)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

datasets = ['HMDB', 'UCF', 'EPIC']
batch_sizes = [32] #[32, 64]#, 128]
learning_rates = [0.1, 0.2, 0.5, 1, 2, 5]
nb_epochs = 1000
results = {}

with open("data_near.pkl", 'rb') as f:
    data_all = pickle.load(f)

for dataset in datasets:
    results[dataset] = {}
    for batch_size in batch_sizes:
        results[dataset][batch_size] = []
        data = data_all[dataset]

        X_tensor_train = torch.tensor(data['X_train'], dtype=torch.float32)
        y_tensor_train = torch.tensor(data['y_train'], dtype=torch.float32)
        train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        X_tensor_test = torch.tensor(data['X_test'], dtype=torch.float32)
        y_tensor_test = torch.tensor(data['y_test'], dtype=torch.float32)
        y_tensor_test = 1-y_tensor_test# pour classification binaire
        test_dataset = TensorDataset(X_tensor_test, y_tensor_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        unique_values, counts = np.unique(y_tensor_train, return_counts=True) ### Entrainement avec pondération
        pos_weight = counts[0] / counts[1] 

        input_dim = 4 #react sur tous + react sur aucun + react video + react flow + react audio
        hidden_dim = 2
        output_dim = 1
        model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)) # Pondération des classes

        for lr in learning_rates:
            optim = torch.optim.SGD(model.parameters(), lr)
            plot_loss, plot_acc, accuracy_train = train(model, train_dataloader, criterion, optim, nb_epochs)
            results[dataset][batch_size].append(accuracy_train)
            #visualize(plot_loss, plot_acc)
            
with open("results_near2.pkl", "wb") as f:
    pickle.dump(results, f)
    
for dataset in datasets:
    for batch_size in batch_sizes:
        plt.plot(learning_rates, results[dataset][batch_size])
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.title(f"Résultats {dataset} - batch_size {batch_size}")
    plt.savefig(f"results_{dataset}_bs{batch_size}.png")
    plt.clf()  # Clear the figure to avoid overlapping plots