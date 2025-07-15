import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

"""
datasets labels: UCF,    HMDB  , EPIC,          HAC (names inside filenames)
datasets names:  UCF101, HMDB51, EPIC-KITCHENS, HAC (dataset directory names)

J'ai fait en sorte que tous les fichiers d'évaluation sauvegardés (les saved_files) aillent dans HMDB-rgb-flow
Ce fichier est censé être chargé dans un script qui va boucler sur les backbones, méthodes, layer_proc, moda_wise ou pas, les datasets etc.
"""

class Dataset():
    def __init__(self, name:str, label:str, ood_mode:str, vfa:str=''):
        """
        backbones_paths: dict[str:str]
        ood_mode: str, "near_ood" ou "far_ood"
        vfa: "vfa" ou "", pour indiquer si on utilise les 3 modalités ou pas. 
        """
        
        assert (vfa and ood_mode == "near_ood") or (not vfa) #vfa implique d'être en near_ood
        
        self.name = name
        self.label = label
        self.ood_mode = ood_mode
        self.vfa = vfa
        
    def backbone_path(self, backbone_type:str) -> str:
        if self.ood_mode == "far_ood" or self.label != "EPIC":
            prefix = "HMDB"
        else:
            prefix = "EPIC"
        #prefix = "EPIC" if self.label == "EPIC" else "HMDB"
        dataset = "HMDB" if self.ood_mode == "far_ood" else self.label
            
        return prefix+"-rgb-flow/{}_{}.pt".format(dataset, "_".join([self.ood_mode, self.vfa, backbone_type]).replace("__","_"))


def get_backbone_path(dataset, ood_mode, backbone_type):
    if dataset == "EPIC":
        return "EPIC-rgb-flow/{}_{}.pt".format(dataset, ood_mode+"_"+backbone_type)
    else:
        return "HMDB-rgb-flow/HMDB_{}.pt".format(dataset, ood_mode+"_"+backbone_type)

#######################################################

FAR_OOD_DATASETS = [Dataset("UCF101", "UCF", "far_ood"), #Dataset("HMDB51", "HMDB", "far_ood")
                    Dataset("EPIC-KITCHENS", "EPIC", "far_ood"), Dataset("HAC", "HAC", "far_ood")]

near_ood_datasets = [Dataset("HMDB51", "HMDB", "near_ood"), Dataset("UCF101", "UCF", "near_ood"),
                     Dataset("EPIC-KITCHENS", "EPIC", "near_ood")] #

vfa_dataset = Dataset("EPIC-KITCHENS", "EPIC", "near_ood", "vfa") 

datasets = {'far_ood': FAR_OOD_DATASETS, 'near_ood':near_ood_datasets, "vfa": vfa_dataset}

#######################################################

def get_y(dataset_id, dataset_ood):
    """
    ID = 0
    OOD = 1
    """
    y_id = np.zeros(dataset_id.shape[0])
    y_id = y_id.reshape(y_id.shape[0], 1)

    y_ood = np.ones(dataset_ood.shape[0])
    y_ood = y_ood.reshape(y_ood.shape[0], 1)

    Y = np.vstack([y_id, y_ood])
    return Y


def get_dataloader(X, y, batch_size):
    """
    X et Y: data['X_train"] ou data['x_test] et data[y_train] y_test
    """
    X_tensor_train = torch.tensor(X, dtype=torch.float32)
    y_tensor_train = torch.tensor(y, dtype=torch.float32)
    train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(len(X_tensor_train), "elements", np.unique(y_tensor_train, return_counts=True))
    return train_dataloader
