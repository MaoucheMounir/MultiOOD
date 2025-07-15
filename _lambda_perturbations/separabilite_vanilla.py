import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import argparse

sys.path.append(os.path.abspath('..'))

from mounirood.utils_mounir import calc_perturbations, ponderation_predictor, get_y
from mounirood.framework import FrameworkFactory
from mounirood.eval_functions import  auc_and_fpr_recall

seed = 42  
np.random.seed(seed)

##########################

# def results_report():
#     print(f"{scores_id.mean()=}, {scores_id.var()=}")
#     print(f"{scores_ood.mean()=}, {scores_ood.var()=}")
#     print("AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc.round(4), fpr.round(4)))

#     plt.hist(scores_id, label="ID")
#     plt.hist(scores_ood, label="OOD")
#     frmwrk = args.framework if args.framework != "near_ood" else args.framework+"_"+dataset
#     title = "ponderation_"+frmwrk+"_"+args.layer_proc
#     plt.title(title)
#     plt.legend()
#     plt.savefig(title+".png")
#     plt.close()

#########################

# parser = argparse.ArgumentParser()
# parser.add_argument("--framework", type=str) # "far_ood", "near_ood", "vfa" 
# parser.add_argument("--layer_proc", type=str) # react, ash
# #parser.add_argument("--appen", type=str, default='') #
# parser.add_argument("--dataset", type=str, default='') # "HMDB", "UCF", "EPIC", "HAC" quand je fixerai le bug
# args = parser.parse_args()

frameworks = ["far_ood", "near_ood", "vfa"]
layer_procs = ["react", "ash"]
datasets = ["HMDB", "UCF", "EPIC"]

results = {}

for framework_name in frameworks:
    if framework_name != "far_ood":
        for dataset in datasets:
            framework = FrameworkFactory(framework_name)
            if framework_name == "near_ood":
                dataset_id, dataset_ood = framework.get_confs("id", 'ash')[dataset], framework.get_confs("ood","ash")[dataset]
            else:
                dataset_id, dataset_ood = framework.get_confs("id", 'ash'), framework.get_confs("ood","ash")
            
            vecteur_sans_id = dataset_id[:, 0] 
            vecteur_sans_ood = dataset_ood[:, 0] 
            results[framework_name+"_"+dataset] = (vecteur_sans_id, vecteur_sans_ood)
                
    else:
        framework = FrameworkFactory("far_ood")
        dataset_id, dataset_ood = framework.get_confs("id", "ash"), framework.get_confs("ood","ash")
        
        vecteur_sans_id = dataset_id[:, 0] 
        vecteur_sans_ood = dataset_ood[:, 0] 
    results[framework_name] = (vecteur_sans_id, vecteur_sans_ood)
    
# Création d'une figure avec 5 lignes, 2 colonnes
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
axes = axes.flatten()  # aplatir en liste 1D

# Boucle d'affichage
for ax, (framework_name, (msp_id, msp_ood)) in zip(axes, results.items()):
    
    ax.hist(msp_id, label="ID")
    ax.hist(msp_ood, label="OOD")
    frmwrk = framework_name if framework_name != "near_ood" else framework_name+"_"+dataset
    title = "vanilla_"+frmwrk
    avg_id, avg_ood = np.round(msp_id.mean(), 2), np.round(msp_ood.mean(), 2)
    ax.set_title(f"{title}, {avg_id=}, {avg_ood=}")

plt.tight_layout()
plt.savefig("separabilite_vanilla2.png")
plt.show()


