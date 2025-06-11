import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from torch import sigmoid 
sys.path.append(os.path.abspath('..'))
from utils_mounir import calc_perturbations
import pandas as pd
from config_mounir import FarOODFramework, NearOODFramework
from IPython.display import display

def calc_acc(outputs, batch_y):
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
    if not isinstance(batch_y, torch.Tensor):
        batch_y = torch.as_tensor(batch_y)
        
    logits = sigmoid(outputs)
    preds = (logits > 0.5).int() # C'est plus efficace avec pytorch pour calculer l'accuracy que plus haut
    acc = (preds == batch_y).float().mean().item()
    return acc

def train(model, train_dataloader, criterion, optim, nb_epochs, scheduler=None):
    plot_loss = []
    plot_acc = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)  
    
    for _ in range(nb_epochs):
        loss_values = []
        acc_values = []
        for i, (batch_X, batch_y) in enumerate(train_dataloader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X) # Retourne l'output du module linéaire. Il faut faire passer par sigmoide
            
            ## preds = np.where(nn.functional.sigmoid(outputs).detach().numpy()>0.5
            ##                , 1, 0)
            
            # logits = sigmoid (outputs)
            # preds = (logits > 0.5).int() # C'est plus efficace avec pytorch pour calculer l'accuracy que plus haut
            # acc = (preds == batch_y).float().mean().item()

            #acc = np.sum((batch_y.numpy() == preds)) / batch_y.shape[0]
            acc = calc_acc(outputs, batch_y)
            acc_values.append(acc)
            
            loss = criterion(outputs, batch_y)
            loss_values.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()
        plot_loss.append(np.mean(loss_values))
        plot_acc.append(np.mean(acc_values))

        if scheduler is not None:
            scheduler.step()
        
    accuracy_train = plot_acc[-1]
    print("accuracy train: ", accuracy_train)
    
    return plot_loss, plot_acc, accuracy_train

def visualize(plot_loss, plot_acc):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    ax[0].plot(np.arange(len(plot_loss)), plot_loss)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Evolution de la loss en fonction des epochs")

    ax[1].plot(np.arange(len(plot_acc)), plot_acc)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Evolution de l'accuracy en fonction des epochs")

    plt.tight_layout()
    plt.show()

def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    #ood_indicator = np.zeros_like(label)
    #ood_indicator[label == -1] = 1
    ood_indicator = label
    
    
    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    
    # Ici, j'ai inversé les signes des 3 confs suivants
    # Parce que dans le classifieur, comme la classe OOD est 1, plus le "score" / logit
    # d'un exemple est élevé, plus il sera classifié comme OOD. Il faut donc inverser le paradigme.
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, -conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr

def calc_outputs_test(model, test_dataloader):
    all_outputs_test = []
    all_labels_test = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  
    
    for batch_X, batch_y in test_dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            outputs = sigmoid(model(batch_X))
        
        all_outputs_test += [x.item() for x in outputs]
        all_labels_test += [x.item() for x in batch_y]
        
        
    logits_id_test = []
    logits_ood_test = []

    for value, label in list(zip(all_outputs_test, all_labels_test)):
        if label == 1:
            logits_ood_test.append(value)
        else:
            logits_id_test.append(value)
            
    outputs_test = {"all_labels_test": all_labels_test,
                "all_outputs_test": all_outputs_test,
                "logits_id_test": logits_id_test,
                "logits_ood_test": logits_ood_test}
    
    return outputs_test


def separabilite(outputs_test):
    
    plt.figure(figsize=(8, 5))

    plt.hist(outputs_test["logits_id_test"], bins=30, alpha=0.6, label="ID", color='blue', density=True)
    plt.hist(outputs_test["logits_ood_test"], bins=30, alpha=0.6, label="OOD", color='red', density=True)

    plt.xlabel("Logits")
    plt.ylabel("Densité")
    plt.title("Distributions de logits ID vs OOD")
    plt.legend()
    plt.show()


def calc_correlations(framework, layer_proc, poids):
    """
    ordre poids: sans, tout, video, flow, [audio]
    """
    poids = list(poids[1:])
    
    # id
    if framework.ood_mode != "near_ood":
        conf_sans, *confs_avec = framework.get_confs("id", layer_proc).transpose()
    else:
        confs:dict = framework.get_confs("id", layer_proc)
        all_confs = []
        for ds, conf_ds in confs.items():
            all_confs.append(conf_ds)
        all_confs = np.vstack(all_confs)
        conf_sans, *confs_avec = all_confs.transpose()        
    
    diffs_id = [np.round(calc_perturbations(conf_sans, conf, "diff"), 4) for conf in confs_avec]


    #ood
    if framework.ood_mode != "near_ood":
        conf_sans, *confs_avec = framework.get_confs("ood", layer_proc).transpose()
    else:
        confs:dict = framework.get_confs("ood", layer_proc)
        all_confs = []
        for ds, conf_ds in confs.items():
            all_confs.append(conf_ds)
        all_confs = np.vstack(all_confs)
        conf_sans, *confs_avec = all_confs.transpose()        
   
   
    diffs_ood = [np.round(calc_perturbations(conf_sans, conf, "diff"), 4) for conf in confs_avec]
    
    if framework.ood_mode != "vfa":
        df = pd.DataFrame([["poids_"+framework.ood_mode]+poids, ["perturbation_"+framework.ood_mode+"_id"]+diffs_id,  ["perturbation_"+framework.ood_mode+"_ood"]+diffs_ood], columns = ["config", 'tout', 'video', 'flow'])
    else:
        df = pd.DataFrame([["poids_"+framework.ood_mode]+poids, ["perturbation_"+framework.ood_mode+"_id"]+diffs_id,  ["perturbation_"+framework.ood_mode+"_ood"]+diffs_ood], columns = ["config", 'tout', 'video', 'flow', 'audio'])

    correlation_id = np.corrcoef(poids, diffs_id)[0,1]
    correlation_ood = np.corrcoef(poids, diffs_ood)[0,1]
    
    return df, correlation_id, correlation_ood

def performance_report(model, test_dataloader, framework, correlations=True):
    """
    correlations (bool): On ne calcule pas la correlation pour le modèle à 2 couches
    """
    outputs_test = calc_outputs_test(model, test_dataloader)
    
    accuracy_test = calc_acc(outputs_test["all_outputs_test"], outputs_test["all_labels_test"])
    print("Accuracy test: ", np.round(accuracy_test, 4))
    
    print("Séparabilité:")
    separabilite(outputs_test)
    
    auroc, _, _, fpr = auc_and_fpr_recall(np.array(outputs_test["all_outputs_test"]), np.array(outputs_test["all_labels_test"]), tpr_th=0.95)
    print("AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc.round(4), fpr.round(4)))
    
    if correlations:
        poids = dict(model.named_parameters())["weight"].cpu().detach().numpy().squeeze().round(4)
        df, correlation_id, correlation_ood = calc_correlations(framework, "react", poids)
        print("Correlation ID : ", correlation_id)
        print("Correlation OOD : ", correlation_ood)
        display(df) 
    else:
        df = correlation_id = correlation_ood = None
        
    return outputs_test, auroc, fpr, df, correlation_id, correlation_ood