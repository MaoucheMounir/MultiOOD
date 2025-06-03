import numpy as np
import torch
from torch.utils.data import  TensorDataset, DataLoader


root_dir_far = "/data/maouche/MultiOOD/HMDB-rgb-flow/saved_files/"
root_dir_near = "/data/maouche/MultiOOD/HMDB-rgb-flow/saved_files/"
ood_datasets = ['UCF', 'EPIC']#, 'HAC']
modalities_far = ['video', 'flow']
modalities_near = ['video', 'flow', 'audio']

def get_id_data(layer_proc):
    template_id_sans_react =  "id_HMDB_conf_baseline_best_val.npy"
    template_id_react_tout = "id_HMDB_conf_baseline_best_"+layer_proc+"_val.npy"
    template_id_par_modalite = "id_HMDB_conf_baseline_best_"+layer_proc+"_{}_val.npy"

    # Sans react
    conf_sans_react = np.load(root_dir_far+template_id_sans_react)
    conf_sans_react = conf_sans_react.reshape(conf_sans_react.shape[0], 1)

    # React sur tout
    conf_react_tout = np.load(root_dir_far+template_id_react_tout)
    conf_react_tout = conf_react_tout.reshape(conf_react_tout.shape[0], 1)


    # React par modalité
    conf_une_modalite = []
    for modality in modalities_far:
        x = np.load(root_dir_far+template_id_par_modalite.format(modality))
        x = x.reshape(x.shape[0], 1)
        conf_une_modalite.append(x)
    conf_une_modalite = np.hstack(conf_une_modalite)


    dataset_id = np.hstack([conf_sans_react, conf_react_tout, conf_une_modalite]) #(N,4)
    return dataset_id

def get_ood_data(layer_proc):
    template_sans_react =  "id_HMDB_ood_{}_conf_baseline_best_eval.npy"
    template_react_tout = "id_HMDB_ood_{}_conf_baseline_best_"+layer_proc+"_eval.npy"
    template_par_modalite = "id_HMDB_ood_{}_conf_baseline_best_"+layer_proc+"_{}_eval.npy"

    # Sans react
    conf_sans_react = []
    for dataset in ood_datasets:
        x = np.load(root_dir_far+template_sans_react.format(dataset))
        x = x.reshape(x.shape[0], 1)
        conf_sans_react.append(x)
    conf_sans_react = np.vstack(conf_sans_react) #(N,1), avec N = 9603, toutes les vidéos des 3 datasets (selon les filtrages far ood)

    # React sur tout 
    conf_react_tout = []
    for dataset in ood_datasets:
        x = np.load(root_dir_far+template_react_tout.format(dataset))
        x = x.reshape(x.shape[0], 1)
        conf_react_tout.append(x)
    conf_react_tout = np.vstack(conf_react_tout) #(N,1)


    # React par modalité
    conf_par_modalite = []
    for modality in modalities_far:
        conf_une_modalite = []
        for dataset in ood_datasets:
            x = np.load(root_dir_far+template_par_modalite.format(dataset, modality))
            x = x.reshape(x.shape[0], 1)
            conf_une_modalite.append(x)
        conf_une_modalite = np.vstack(conf_une_modalite)
        conf_par_modalite.append(conf_une_modalite)

    conf_par_modalite = np.reshape(conf_par_modalite, (-1, len(modalities_far)))  #(N,1)

    dataset_ood = np.hstack([conf_sans_react, conf_react_tout, conf_par_modalite]) #(N,4)
    return dataset_ood

def get_id_data_near(layer_proc):
    #Near vfa pour le coup
    template_sans_react =  "id_EPIC_near_ood_conf_vfa_baseline_best_test.npy"
    template_react_tout = "id_EPIC_near_ood_conf_vfa_baseline_best_"+layer_proc+"_test.npy"
    template_par_modalite = "id_EPIC_near_ood_conf_vfa_baseline_best_"+layer_proc+"_{}_test.npy"

    # Sans react
    conf_sans_react = np.load(root_dir_near+template_sans_react)
    conf_sans_react = conf_sans_react.reshape(conf_sans_react.shape[0], 1)

    # React sur tout
    conf_react_tout = np.load(root_dir_near+template_react_tout)
    conf_react_tout = conf_react_tout.reshape(conf_react_tout.shape[0], 1)


    # React par modalité
    conf_une_modalite = []
    for modality in modalities_near:
        x = np.load(root_dir_near+template_par_modalite.format(modality))
        x = x.reshape(x.shape[0], 1)
        conf_une_modalite.append(x)
    conf_une_modalite = np.hstack(conf_une_modalite)

    dataset_id = np.hstack([conf_sans_react, conf_react_tout, conf_une_modalite]) #(N,5)
    return dataset_id


def get_ood_data_near(layer_proc):
    #VFA pas juste near
    template_sans_react =  "id_EPIC_near_ood_conf_vfa_baseline_best_eval.npy"
    template_react_tout = "id_EPIC_near_ood_conf_vfa_baseline_best_"+layer_proc+"_eval.npy"
    template_par_modalite = "id_EPIC_near_ood_conf_vfa_baseline_best_"+layer_proc+"_{}_eval.npy"

    # Sans react
    conf_sans_react = np.load(root_dir_near+template_sans_react)
    conf_sans_react = conf_sans_react.reshape(conf_sans_react.shape[0], 1)

    # React sur tout
    conf_react_tout = np.load(root_dir_near+template_react_tout)
    conf_react_tout = conf_react_tout.reshape(conf_react_tout.shape[0], 1)


    # React par modalité
    conf_une_modalite = []
    for modality in modalities_near:
        x = np.load(root_dir_near+template_par_modalite.format(modality))
        x = x.reshape(x.shape[0], 1)
        conf_une_modalite.append(x)
    conf_une_modalite = np.hstack(conf_une_modalite)


    dataset_ood = np.hstack([conf_sans_react, conf_react_tout, conf_une_modalite]) #(N,5)
    return dataset_ood

def get_y(dataset_id, dataset_ood):
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