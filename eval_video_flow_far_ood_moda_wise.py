import numpy as np
from metrics import compute_all_metrics, auc_and_fpr_recall
import torch
import argparse
import faiss
import sklearn.covariance
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from utils_mounir import save_results_gen


def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc

parser = argparse.ArgumentParser()
parser.add_argument("--appen", type=str, default='a2d_npmix_best_') # a2d_npmix_best_ a2d_npmix_best_ash_ a2d_npmix_best_react_
parser.add_argument("--aggregation", type=str, default='max') 
args = parser.parse_args()

num_classes = 43

aggregation_method = {"mean":np.mean, "max": np.max}

ood_datasets = ["UCF", "EPIC"]#, "HAC"]
modalities = ["video", "flow"]
args.path = "HMDB-rgb-flow"
args.dataset = "HMDB"


for dataset in ood_datasets:

    args.ood_dataset = dataset
    confs = []
    id_accs = []
    
    for modality in modalities:
        
        split = 'test'
        print(split)
        pred_name = args.path + '/saved_files/id_'+args.dataset+'_pred_' + args.appen + modality+"_" + split + '.npy'
        conf_name = args.path + '/saved_files/id_'+args.dataset+'_conf_' + args.appen + modality+"_" + split + '.npy'
        label_name = args.path + '/saved_files/id_'+args.dataset+'_label_' + args.appen + modality+"_" + split + '.npy'

        id_pred = np.load(pred_name)
        id_conf = np.load(conf_name)
        id_gt = np.load(label_name)
        
        ID_ACC = acc(id_pred, id_gt)
        id_accs.append(ID_ACC)
        
        split = 'eval'
        print(split)

        conf_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_conf_' + args.appen + modality+"_" + split + '.npy'
        label_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_label_' + args.appen + modality+"_" + split + '.npy'
        ood_gt = np.load(label_name)
        ood_conf = np.load(conf_name)
        confs.append(ood_conf)
            
    # Fuse the modalities
    ID_ACC = aggregation_method[args.aggregation](id_accs)
    
    
    ood_conf = aggregation_method[args.aggregation](np.vstack(confs), axis=0)
    ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
    #pred = np.concatenate([id_pred, ood_pred])
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt, ood_gt])
    
    # Compute the metric
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)
    #ood_metrics = compute_all_metrics(conf, label, pred)

    print("FPR@95: ", fpr)
    print("AUROC: ", auroc)


    save_results_gen(backbone="baseline", method=args.aggregation,
                    dataset=args.ood_dataset, layer_proc=args.appen,
                    fpr95=fpr, auroc=auroc,
                    id_acc=ID_ACC, exec_time="N/A", filename="aggregation_ash_far_ood.csv", appen=args.appen)