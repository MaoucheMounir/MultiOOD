import numpy as np
import pickle
from mounirood.datasets import get_dataloader
import torch
import torch.nn as nn
from mounirood.eval_functions import visualize, train, performance_report

class Classifier():
    def __init__(self, framework, dims:list, ):
        #dims en entrée: ash sur tous + ash sur aucun + ash video + ash flow + ash audio
        self.input_dim, self.output_dim = dims[0], dims[-1]
        if len(dims) == 3:
            self.hidden_dim = dims[1]
        # AJOUTER un argument et attribut pour le chemin du dataset à charger     
        self.framework = framework # Définit si on est en near, far ou vfa, et quel dataset utiliser
        self.data = ""    
    
    def load_data(self):
        data_filenames = {"near_ood":"data_near.pkl", "far_ood": "data_vf.pkl", "vfa":"data.vfa"}
        
        with open(data_filenames.get(self.framework.ood_mode, -1), 'rb') as f:#"/data/maouche/MultiOOD/_failure_prediction/fp_ash/data_near.pkl", 'rb') as f:
            data = pickle.load(f)    
        if self.framework.ood_mode == "near_ood":
            self.data = data[self.framework.dataset_used]
            
        self.train_dataloader = get_dataloader(self.data["X_train"], self.data["y_train"], batch_size=128) 
        self.test_dataloader = get_dataloader(self.data["X_test"], self.data["y_test"], batch_size=128) 
        #watxh out

    def calc_class_weight(self, ):
        if not self.data:
            self.load_data()
        
        # Calcul des poids des classes
        y_tensor_train = torch.tensor(self.data['y_train'], dtype=torch.float32)
        unique_values, counts = np.unique(y_tensor_train, return_counts=True) ### Entrainement avec pondération
        #ratio = counts/np.sum(counts)
        self.pos_weight = counts[0] / counts[1] # N_negative / N_positive
    
    
    def train_models(self, learning_rates=[1e-1, 1, 2]):
        if not self.data:
            self.load_data()
        if not self.pos_weight:
            self.calc_class_weight()
        
        nb_epochs = 1000
        models = {}
        for lr in learning_rates :
            model = nn.Linear(self.input_dim, self.output_dim)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight)) # Pondération des classes
            optim = torch.optim.SGD(model.parameters(), lr)

            plot_loss, plot_acc, accuracy_train = train(model, self.train_dataloader, criterion, optim, nb_epochs)
            visualize(plot_loss, plot_acc)
            models[model] = accuracy_train
            
        self.models = models
        
    def save_best_model(self, best_model_idx=None):    
        self.best_model = sorted(list(self.models.items()), key=lambda x:x[1], reverse=True)[0][0] if not best_model_idx else self.moels[best_model_idx]
        torch.save(self.best_model.state_dict(), f"model_{self.framework.ood_mode}_{self.framework.dataset_used}.pth")
    
    def performance_report(self, model_name=""):
        if self.best_model :
            model = self.best_model 
        else:
            model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.output_dim))
            if not model_name:
                model_name = f"model_{self.framework.ood_mode}_{self.framework.dataset_used}.pth"
            model.load_state_dict(torch.load(model_name))
        
        return performance_report(model, self.test_dataloader, self.framework)
        