import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from utils_mounir import calc_perturbations, get_y
from eval_functions import auc_and_fpr_recall

class PonderationTester():
    def __init__(self, framework, layer_proc, dataset=""):
        assert (framework.ood_mode == "near_ood") == bool(dataset), "Si far_ood ou vfa, tous les datasets sont confondus"
        #framework de type Framework
        self.framework = framework
        self.layer_proc = layer_proc
        self.dataset = dataset
    
    def calc_scores(self, layer_proc, method):
        """
        method = ponderation_predictor, 
        """
        if self.framework.ood_mode == "near_ood":
            dataset_id, dataset_ood = self.framework.get_confs("id", layer_proc)[self.dataset], self.framework.get_confs("ood", layer_proc)[self.dataset]
        else:
            dataset_id, dataset_ood = self.framework.get_confs("id", layer_proc), self.framework.get_confs("ood", layer_proc)
        self.dataset_id, self.dataset_ood = dataset_id, dataset_ood
        
        vecteur_sans_id = dataset_id[:, 0] 
        confs_modalites_id = dataset_id[:, 2:]
        perturbations_id = np.array([calc_perturbations(vecteur_sans_id, vecteur_avec) for vecteur_avec in confs_modalites_id.transpose()])
        # shape : [N_modalités,]

        vecteur_sans_ood = dataset_ood[:, 0] 
        confs_modalites_ood = dataset_ood[:, 2:]
        perturbations_ood = np.array([calc_perturbations(vecteur_sans_ood, vecteur_avec) for vecteur_avec in confs_modalites_ood.transpose()])
        # shape : [N_modalités,]

        scores_id:np.ndarray = method(confs_modalites_id, perturbations_id)
        scores_ood:np.ndarray = method(confs_modalites_ood, perturbations_ood)
        
        return scores_id, scores_ood
        
    def get_scores(self, method):
        scores = defaultdict(dict)
        
        if self.layer_proc == "both":
            scores_id_react, scores_ood_react = self.calc_scores("react", method)    
            scores_id_ash, scores_ood_ash = self.calc_scores("ash", method)    
            scores["react"]["id"], scores["react"]["ood"] = scores_id_react, scores_ood_react
            scores["ash"]["id"], scores["ash"]["ood"] = scores_id_ash, scores_ood_ash
        
        else:
            scores_id, scores_ood = self.calc_scores(self.layer_proc, method)    
            scores["id"], scores["ood"] = scores_id, scores_ood 
        
        self.scores = scores
        return scores
    
    def separabilite(self, scores, plot_id=True, plot_ood=True):
        if self.layer_proc != "both":
            scores_id, scores_ood = scores["id"], scores["ood"]
            print(f"{scores_id.mean()=}, {scores_id.var()=}")
            print(f"{scores_ood.mean()=}, {scores_ood.var()=}")

            scores = np.vstack((scores_id, scores_ood))
            labels = get_y(self.dataset_id, self.dataset_ood)
            auroc, _, _, fpr = auc_and_fpr_recall(scores, labels, tpr_th=0.95)
            print("AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc.round(4), fpr.round(4)))

            if plot_id:
                plt.hist(scores_id, label="ID", histtype="step", linewidth=2)
            if plot_ood:
                plt.hist(scores_ood, label="OOD", alpha=0.85)
            plt.title(f"{self.framework.ood_mode}_{self.layer_proc}_{self.dataset}")
            plt.legend()
        
        else:
            scores_id_react, scores_ood_react = scores["react"]["id"], scores["react"]["ood"]
            scores_id_ash, scores_ood_ash =  scores["ash"]["id"], scores["ash"]["ood"]
            
            #print(f"{.mean()=}, {scores_id.var()=}")
            #print(f"{scores_ood.mean()=}, {scores_ood.var()=}")

            scores_react = np.vstack((scores_id_react, scores_ood_react))
            scores_ash = np.vstack((scores_id_ash, scores_ood_ash))
            
            labels = get_y(self.dataset_id, self.dataset_ood)
            auroc_react, _, _, fpr_react = auc_and_fpr_recall(scores_react, labels, tpr_th=0.95)
            auroc_ash, _, _, fpr_ash = auc_and_fpr_recall(scores_ash, labels, tpr_th=0.95)
            print("ReAct AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc_react.round(4), fpr_react.round(4)))
            print("ASh AUC ROC: {}\nFPR@TPR95: {}\n".format(auroc_ash.round(4), fpr_ash.round(4)))

            if plot_id:
                plt.hist(scores_id_react, label="ID ReAct")
                plt.hist(scores_id_ash, label="ID ASh")
            if plot_ood:
                plt.hist(scores_ood_react, label="OOD ReAct")
                plt.hist(scores_ood_ash, label="OOD ASh")
                
            plt.title(f"{self.framework.ood_mode}_react_vs_ash_{self.dataset}")
            plt.legend()
        
    def test_report(self, method, plot_id=True, plot_ood=True):
        self.get_scores(method)
        self.separabilite(self.scores, plot_id, plot_ood)
        