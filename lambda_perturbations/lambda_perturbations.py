import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import argparse

#sys.path.append(os.path.abspath('..'))

from utils_mounir import calc_perturbations, ponderation_predictor, get_y
from config_mounir import FarOODFramework, NearOODFramework
from eval_functions import  auc_and_fpr_recall

seed = 42  
np.random.seed(seed)

##########################

def save_results():
    frmwrk = args.framework if args.framework != "near_ood" else args.framework+"_"+dataset
    columns = ["method", "framework", "layer_proc", "auroc", "fpr"] 
    info = f"ponderation_perturbations,{frmwrk},{args.layer_proc},{auroc},{fpr}"
    
    filename = "resultats_ponderations.csv"
    
    if not os.path.exists(filename): 
        with open(filename, "w") as f:
            f.write(",".join(columns)+"\n")
            f.write(info+'\n')
    else:
        with open(filename, "a") as f:
            f.write(info+'\n')

def results_report():
    print(f"{scores_id.mean()=}, {scores_id.var()=}")
    print(f"{scores_ood.mean()=}, {scores_ood.var()=}")
    print("AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc.round(4), fpr.round(4)))

    plt.hist(scores_id, label="ID")
    plt.hist(scores_ood, label="OOD")
    frmwrk = args.framework if args.framework != "near_ood" else args.framework+"_"+dataset
    title = "ponderation_"+frmwrk+"_"+args.layer_proc
    plt.title(title)
    plt.legend()
    plt.savefig(title+".png")
    plt.close()

#########################

parser = argparse.ArgumentParser()
parser.add_argument("--framework", type=str) # "far_ood", "near_ood", "vfa" 
parser.add_argument("--layer_proc", type=str) # react, ash
#parser.add_argument("--appen", type=str, default='') #
parser.add_argument("--dataset", type=str, default='') # "HMDB", "UCF", "EPIC", "HAC" quand je fixerai le bug
args = parser.parse_args()

near_ood = args.framework == "near_ood"
dataset = args.dataset

#assert (not near_ood) or (dataset != '')
assert (near_ood and dataset != '') or (not near_ood and dataset == '')
 
if args.framework == "far_ood":
    framework = FarOODFramework()
else:
    framework = NearOODFramework(args.framework)


if near_ood: 
    # Charger les données avec perturbations
    dataset_id, dataset_ood = framework.get_confs("id", args.layer_proc)[dataset], framework.get_confs("ood",args.layer_proc)[dataset]
else:
    dataset_id, dataset_ood = framework.get_confs("id", args.layer_proc), framework.get_confs("ood",args.layer_proc)

vecteur_sans_id = dataset_id[:, 0] 
confs_modalites_id = dataset_id[:, 2:]
perturbations_id = np.array([calc_perturbations(vecteur_sans_id, vecteur_avec) for vecteur_avec in confs_modalites_id.transpose()])
# shape : [N_modalités,]

vecteur_sans_ood = dataset_ood[:, 0] 
confs_modalites_ood = dataset_ood[:, 2:]
perturbations_ood = np.array([calc_perturbations(vecteur_sans_ood, vecteur_avec) for vecteur_avec in confs_modalites_ood.transpose()])
# shape : [N_modalités,]

scores_id = ponderation_predictor(confs_modalites_id, perturbations_id)
scores_ood = ponderation_predictor(confs_modalites_ood, perturbations_ood)

scores = np.vstack((scores_id, scores_ood))
labels = get_y(dataset_id, dataset_ood)
auroc, _, _, fpr = auc_and_fpr_recall(scores, labels, tpr_th=0.95)

#print("AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc.round(4), fpr.round(4)))
# save_results()

results_report()