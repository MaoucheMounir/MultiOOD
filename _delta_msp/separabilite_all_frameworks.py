import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import argparse

sys.path.append(os.path.abspath('..'))

from mounirood.modality_fusion import calc_perturbations
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
methods = ["diff_vector_abs", "diff_vector"]
results = {}

for method in methods:
    for framework_name in frameworks:
        for layer_proc in layer_procs:
            if framework_name == "near_ood":
                for dataset in datasets:
                    framework = FrameworkFactory(framework_name)
                    if framework_name == "near_ood":
                        dataset_id, dataset_ood = framework.get_confs("id", layer_proc)[dataset], framework.get_confs("ood", layer_proc)[dataset]
                    else:
                        dataset_id, dataset_ood = framework.get_confs("id", layer_proc), framework.get_confs("ood", layer_proc)
                    
                    vecteur_sans_id = dataset_id[:, 0] 
                    confs_modalites_id = dataset_id[:, 2:]
                    
                    perturbations_id = np.array([-calc_perturbations(vecteur_sans_id, vecteur_avec, method=method) for vecteur_avec in confs_modalites_id.transpose()]).transpose()

                    vecteur_sans_ood = dataset_ood[:, 0] 
                    confs_modalites_ood = dataset_ood[:, 2:]
                    perturbations_ood = np.array([-calc_perturbations(vecteur_sans_ood, vecteur_avec, method=method) for vecteur_avec in confs_modalites_ood.transpose()]).transpose()
                    
                    scores_id = np.sum(perturbations_id, axis=1).reshape(-1,1)
                    scores_ood = np.sum(perturbations_ood, axis=1).reshape(-1,1)
                    
                    results[framework_name+"_"+dataset+'_'+layer_proc] = (scores_id, scores_ood)
                        
            else:
                framework = FrameworkFactory(framework_name)
                dataset_id, dataset_ood = framework.get_confs("id", layer_proc), framework.get_confs("ood",layer_proc)
                
                vecteur_sans_id = dataset_id[:, 0] 
                confs_modalites_id = dataset_id[:, 2:]
                # delta_msp
                perturbations_id = np.array([-calc_perturbations(vecteur_sans_id, vecteur_avec, method=method) for vecteur_avec in confs_modalites_id.transpose()]).transpose()
                # shape : [N_modalités,]

                vecteur_sans_ood = dataset_ood[:, 0] 
                confs_modalites_ood = dataset_ood[:, 2:]
                # delta_msp
                perturbations_ood = np.array([-calc_perturbations(vecteur_sans_ood, vecteur_avec, method=method) for vecteur_avec in confs_modalites_ood.transpose()]).transpose()

                
                scores_id = np.sum(perturbations_id, axis=1).reshape(-1,1)
                scores_ood = np.sum(perturbations_ood, axis=1).reshape(-1,1)
                results[framework_name+'_'+layer_proc] = (scores_id, scores_ood)
        
    # Création d'une figure avec 5 lignes, 2 colonnes
    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    axes = axes.flatten()  # aplatir en liste 1D

    # Boucle d'affichage
    for ax, (framework_name, (msp_id, msp_ood)) in zip(axes, results.items()):
        
        ax.hist(msp_id, label="ID", histtype="step", linewidth=2)
        ax.hist(msp_ood, label="OOD", alpha=0.85)
        frmwrk = framework_name if framework_name != "near_ood" else framework_name+"_"+dataset
        title = frmwrk
        avg_id, avg_ood = np.round(msp_id.mean(), 2), np.round(msp_ood.mean(), 2)
        ax.set_title(f"{title}, {avg_id=}, {avg_ood=}")
        ax.legend()
        

    plt.tight_layout()
    plt.savefig("separabilite_delta_msp_"+method+".png")
    plt.show()


