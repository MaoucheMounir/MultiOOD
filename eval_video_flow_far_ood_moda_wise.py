import numpy as np
from metrics import auc_and_fpr_recall
import argparse
from mounirood.utils_mounir import save_results_gen, max_perturbations, create_file


def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc

parser = argparse.ArgumentParser()
parser.add_argument("--appen", type=str, default='baseline_best_') # a2d_npmix_best_ a2d_npmix_best_ash_ a2d_npmix_best_react_
#parser.add_argument("--aggregation", type=str, default='max') 
args = parser.parse_args()

num_classes = 43

aggregation_methods = ["mean", "max", "min", "max_perturbation"]

def aggregate(values, method):
    """
    values: liste de vecteurs. 
    Les fonctions d'agrégation reçoivent ces vecteurs empilés en lignes
    """
    values = np.vstack(values)
    if method == "mean":
        return np.mean(values, axis=0)
    elif method == "max":
        return np.max(values, axis=0)
    elif method == "min":
        return np.min(values, axis=0)
    elif method == "max_perturbation":
        #tmp = args.path + '/saved_files/id_'+args.dataset+'_near_ood_conf_' + args.appen.replace("ash_", "").replace("react_", "")  + "test" + '.npy'
        tmp = args.path + '/saved_files/id_'+args.dataset+'_conf_' + args.appen.replace("ash_", "").replace("react_", "") + "test" + '.npy'
        id_conf_sans_react = np.load(tmp)
        return max_perturbations(id_conf_sans_react, id_confs, values)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


ood_datasets = ["UCF", "EPIC"]#, "HAC"]
modalities = ["video", "flow"]
args.path = "HMDB-rgb-flow"
args.dataset = "HMDB"

# Creer le fichier et remplir l'en tete
filename = create_file("aggregation_far_", args.appen) 

for aggregation in aggregation_methods:
    for dataset in ood_datasets:

        args.ood_dataset = dataset
        ood_confs = []
        id_accs = []
        id_confs = []
        
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
            id_confs.append(id_conf)
            
            split = 'eval'
            print(split)

            conf_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_conf_' + args.appen + modality+"_" + split + '.npy'
            label_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_label_' + args.appen + modality+"_" + split + '.npy'
            ood_gt = np.load(label_name)
            ood_conf = np.load(conf_name)
            ood_confs.append(ood_conf)
                
        # Fuse the modalities
        ID_ACC = aggregate(id_accs, "mean").item()
        id_conf = aggregate(id_confs, aggregation)
        
        ood_conf = aggregate(ood_confs, aggregation)
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


        save_results_gen(backbone="baseline", method=aggregation,
                        dataset=dataset, layer_proc=args.appen,
                        fpr95=fpr, auroc=auroc,
                        id_acc=ID_ACC, exec_time="N/A", filename=filename, appen=args.appen)