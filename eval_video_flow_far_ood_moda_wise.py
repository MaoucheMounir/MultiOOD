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


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
            
    def forward(self, afeat, ffeat):
        feat = torch.cat((afeat, ffeat), dim=1)
        return self.enc_net(feat)

def generalized_entropy(softmax_id_val, gamma=0.1, M=20):
        probs =  softmax_id_val 
        probs_sorted = np.sort(probs, axis=1)[:,-M:]
        scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
        return -scores 

def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc

#normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

parser = argparse.ArgumentParser()
#parser.add_argument("--postprocessor", type=str, default='msp') # 'msp' 'ebo' 'maxlogit' 'Mahalanobis' 'ash' 'react' 'knn' 'gen' 'vim'
parser.add_argument("--appen", type=str, default='a2d_npmix_best_') # a2d_npmix_best_ a2d_npmix_best_ash_ a2d_npmix_best_react_
parser.add_argument("--aggregation", type=str, default='max') 
#parser.add_argument("--dataset", type=str, default='HMDB') # HMDB Kinetics
#parser.add_argument("--ood_dataset", type=str, default='UCF') # HMDB UCF Kinetics EPIC HAC
#parser.add_argument("--path", type=str, default='HMDB-rgb-flow') # HMDB-rgb-flow EPIC-rgb-flow
#parser.add_argument("--resume_file", type=str, default='HMDB-rgb-flow/models/checkpoint.pt') # for vim 'HMDB_far_ood_a2d_npmix.pt'
args = parser.parse_args()

num_classes = 43

aggregation_method = {"mean":np.mean, "max": np.max}

ood_datasets = ["UCF", "EPIC", "HAC"]
modalities = ["video", "flow"] #"sound"
args.path = "HMDB-rgb-flow"
args.dataset = "HMDB"

print("boucle0")
for dataset in ood_datasets:
    
    

    # if args.postprocessor == 'ebo' or args.postprocessor == 'ash' or args.postprocessor == 'react':
    #     temperature = 1.0
    #     id_output = torch.tensor(id_output)
    #     id_conf = temperature * torch.logsumexp(id_output / temperature, dim=1)
    #     id_conf = id_conf.numpy()
    args.ood_dataset = dataset
    confs = []
    id_accs = []
    
    print("boucle1")
    for modality in modalities:
        
        split = 'test'
        print(split)
        #output_name = args.path + '/saved_files/id_'+args.dataset+'_output_' + args.appen + split + '.npy'
        pred_name = args.path + '/saved_files/id_'+args.dataset+'_pred_' + args.appen + split + '.npy'
        conf_name = args.path + '/saved_files/id_'+args.dataset+'_conf_' + args.appen + split + '.npy'
        label_name = args.path + '/saved_files/id_'+args.dataset+'_label_' + args.appen + split + '.npy'
        #feature_name = args.path + '/saved_files/id_'+args.dataset+'_feature_' + args.appen + split + '.npy'
        #id_output = np.load(output_name)
        id_pred = np.load(pred_name)
        id_conf = np.load(conf_name)
        id_gt = np.load(label_name)
        #id_feature = np.load(feature_name)

        ID_ACC = acc(id_pred, id_gt)
        id_accs.append(ID_ACC)
        
        split = 'eval'
        print(split)

        #output_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_output_' + args.appen + split + '.npy'
        #pred_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_pred_' + args.appen + split + '.npy'
        conf_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_conf_' + args.appen + modality+"_" + split + '.npy'
        label_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_label_' + args.appen + modality+"_" + split + '.npy'
        #feature_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_feature_' + args.appen + split + '.npy'
        #ood_output = np.load(output_name)
        #ood_pred = np.load(pred_name)
        ood_gt = np.load(label_name)
        ood_conf = np.load(conf_name)
        confs.append(ood_conf)
            
        #ood_feature = np.load(feature_name)

        # if args.postprocessor == 'ebo' or args.postprocessor == 'ash' or args.postprocessor == 'react':
        #     ood_output = torch.tensor(ood_output)
        #     ood_conf = temperature * torch.logsumexp(ood_output / temperature, dim=1)
        #     ood_conf = ood_conf.numpy()
    print("boucle2")
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
                    id_acc=ID_ACC, exec_time="N/A", filename="results_moda_wise_ucf_react_video_FINAL.csv", appen=args.appen)