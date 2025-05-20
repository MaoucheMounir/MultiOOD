import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn

def train(model, train_dataloader, criterion, optim, nb_epochs):
    plot_loss = []
    plot_acc = []

    for _ in range(nb_epochs):
        loss_values = []
        acc_values = []
        for i, (batch_X, batch_y) in enumerate(train_dataloader):
            outputs = model(batch_X) # Retourne l'output du module linéaire. Il faut faire passer par sigmoide
            
            #preds = np.where(nn.functional.sigmoid(outputs).detach().numpy()>0.5
            #                , 1, 0)
            logits = nn.functional.sigmoid(outputs)
            preds = (logits > 0.5).int() # C'est plus efficace avec pytorch pour calculer l'accuracy que plus haut
            acc = (preds == batch_y).float().mean().item()

            #acc = np.sum((batch_y.numpy() == preds)) / batch_y.shape[0]
            acc_values.append(acc)
            
            loss = criterion(outputs, batch_y)
            loss_values.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()
        plot_loss.append(np.mean(loss_values))
        plot_acc.append(np.mean(acc_values))

    accuracy_train = np.mean(plot_acc)
    print("accuracy train: ", accuracy_train)
    
    




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

    for batch_X, batch_y in test_dataloader:
        with torch.no_grad():
            outputs = nn.functional.sigmoid(model(batch_X))
        
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