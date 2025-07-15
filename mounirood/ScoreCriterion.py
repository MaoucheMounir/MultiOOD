from abc import ABC
from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath('..'))
import torch
from sklearn import metrics

from mounirood.tester import PonderationTester
from mounirood.datasets import get_y
from mounirood.framework import FrameworkFactory
from mounirood.modality_fusion import calc_perturbations, CoefficientPonderator
from mounirood.eval_functions import auc_and_fpr_recall, visualize, calc_outputs_test, calc_correlations, separabilite
from mounirood.PerturbationCalculator import PerturbationCalculator


"""
Les scores finaux doivent être élevés pour les ID et faibles pour OOD.
En termes de scores, ID -> 1, OOD -> O
En termes de classes, ID = 0, OOD = 1. C'est pour ça que dans la fonction auc_and_fpr du papier ils multiplient par -1 le 'conf'. 
"""

class ScoreCriterion(ABC):
    def __init__(self):
        self.perturbation_method = None
        pass

    def __call__(self, dataset_id, dataset_ood):
        pass
    
    def get_perturbations(self, dataset_id, dataset_ood) -> Tuple[np.ndarray, np.ndarray]:
        """
        Partie commune à tous les critères
        Returns:
            scores_id (np.ndarray): Tableau numpy de forme (N,)
            scores_ood (np.ndarray): Tableau numpy de forme (N,)
        """
        # Faire un truc avec les confs
        
        vecteur_sans_id = dataset_id[:, 0] 
        confs_modalites_id = dataset_id[:, 2:] # (N, M). Ensuite on transpose parce que le vecteur_sans est (N,)
        vecteur_sans_ood = dataset_ood[:, 0] 
        confs_modalites_ood = dataset_ood[:, 2:]
        
        #perturbations_id = np.array([-calc_perturbations(vecteur_sans_id, vecteur_avec, method="diff_vector_abs") for vecteur_avec in confs_modalites_id.transpose()]).transpose()
        perturbations_id = np.array([self.perturbation_method(vecteur_sans_id, vecteur_avec) for vecteur_avec in confs_modalites_id.transpose()]).transpose()
        # shape : [N_modalités,]

        perturbations_ood = np.array([self.perturbation_method(vecteur_sans_ood, vecteur_avec) for vecteur_avec in confs_modalites_ood.transpose()]).transpose()
        # shape : [N_modalités,]
        
        return perturbations_id, perturbations_ood
        
    
    def validate_scores(dataset_id, dataset_ood, scores_id, scores_ood):
        assert len(dataset_id.shape) == len(dataset_ood.shape)
        # Décider si je veux des vecteurs ou des matrices (N,1)
        assert len(scores_id.shape) == len(scores_ood.shape), "Dimension des scores ID et OOD différentes"
        assert scores_id == len(dataset_id), "Erreur nb lignes ID"
        assert scores_ood == len(dataset_id), "Erreur nb lignes OOD"
    
    

class DeltaMSP(ScoreCriterion):
    def __init__(self, use_absolute_value=True):
        """
        Utilise la différence des MSPs (perturbation) directement comme score
        """
        
        self.use_abs = use_absolute_value 
        
        self.perturbation_method = PerturbationCalculator("diff_vector_abs" if self.use_abs else "diff_vector")
        pass
    
    def __call__(self, dataset_id, dataset_ood):
        #return super().__call__(dataset_id, dataset_ood)
        perturbations_id, perturbations_ood = super().get_perturbations(dataset_id, dataset_ood)
        return self.compute_scores(perturbations_id, perturbations_ood)
    
    
    def compute_scores(self, perturbations_id, perturbations_ood=None):
        """
        # On reshape de cette manière pour qu'on puisse les vstack
        # Avoir des scores à la fin qu'on pourra évaluer directement en AUC et FPR
        # On met au négatif car on veut que les ID aient un score supérieur aux OOD
        """
        
        
        scores_id = -np.sum(perturbations_id, axis=1).reshape(-1,1)
        scores_ood = -np.sum(perturbations_ood, axis=1).reshape(-1,1)
        return scores_id, scores_ood