import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import argparse

sys.path.append(os.path.abspath('..'))

from mounirood.datasets import get_y
from mounirood.modality_fusion import calc_perturbations
from mounirood.framework import FrameworkFactory 
from mounirood.eval_functions import  auc_and_fpr_recall
from mounirood.ScoreCriterion import DeltaMSP
#from metrics import auc_and_fpr_recall 

seed = 42  
np.random.seed(seed)

# Résultats identiques à celui sans _maj

##########################

def save_results():
    frmwrk = args.framework if args.framework != "near_ood" else args.framework+"_"+dataset
    columns = ["method", "framework", "layer_proc", "auroc", "fpr"] 
    info = f"ponderation_perturbations,{frmwrk},{args.layer_proc},{auroc*100},{fpr*100}"
    
    filename = "resultats_delta_msp_sans_abs.csv"
    
    if not os.path.exists(filename): 
        with open(filename, "w") as f:
            f.write(",".join(columns)+"\n")
            f.write(info+'\n')
    else:
        with open(filename, "a") as f:
            f.write(info+'\n')

def performance_report():
    print(f"{scores_id.mean()=}, {scores_id.var()=}")
    print(f"{scores_ood.mean()=}, {scores_ood.var()=}")
    print("AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc.round(4), fpr.round(4)))
    
    plt.hist(scores_id, label="ID", histtype="step", linewidth=2) #
    plt.hist(scores_ood, label="OOD", alpha=0.85)
    
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
parser.add_argument("--save_results", action='store_true')
parser.add_argument("--performance_report", action='store_true')
args = parser.parse_args()

near_ood = args.framework == "near_ood"
dataset = args.dataset

#assert (not near_ood) or (dataset != '')
assert (near_ood and dataset != '') or (not near_ood and dataset == '')
 
framework = FrameworkFactory(args.framework)

if near_ood: 
    # Charger les données avec perturbations
    dataset_id, dataset_ood = framework.get_confs("id", args.layer_proc)[dataset], framework.get_confs("ood",args.layer_proc)[dataset]
else:
    dataset_id, dataset_ood = framework.get_confs("id", args.layer_proc), framework.get_confs("ood",args.layer_proc)

delta_msp = DeltaMSP()
scores_id, scores_ood = delta_msp(dataset_id, dataset_ood)

scores = np.vstack((scores_id, scores_ood))
labels = get_y(dataset_id, dataset_ood)

auroc, _, _, fpr = auc_and_fpr_recall(scores, labels, tpr_th=0.95)

#print("AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc.round(4), fpr.round(4)))

if args.save_results:
    save_results()
if args.performance_report:
    performance_report()