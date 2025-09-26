import numpy as np

def calc_perturbations(conf1:np.ndarray, conf2:np.ndarray, method="mse") -> np.ndarray:
    """
    args:
    conf1, conf2 (np.ndarray, list)
    conf1 est en général le score de base et conf2 avec react
    """
    
    if method == "mse":
        return np.mean((conf1 - conf2) ** 2)
    if method == "diff":
        return np.mean(conf1-conf2)
    if method == "diff_vector":
        return conf2-conf1
    if method == "diff_vector_abs":
        return np.abs(conf2-conf1)
    else:
        raise ValueError("méthode non reconnue")
    
def max_perturbations(id_conf_sans, id_confs, ood_confs):
    """
    id_conf_sans: scores de confiance obtenus sans react/ash
    id_conf, ood_confs: scores de confiance obtenus avec react/ash
    """
    max_perturbations = -1
    max_idx = -1
    
    for i, conf in enumerate(id_confs):
        perturbation = calc_perturbations(id_conf_sans, conf)
        if perturbation > max_perturbations:
            max_perturbations = perturbation
            max_idx = i
    
    assert max_idx != -1 
    
    return ood_confs[max_idx]

def ponderer_perturbations(id_conf_sans, id_confs, ood_confs, fct_ponderation):
    """
    id_conf_sans: scores de confiance obtenus sans react/ash
    id_conf, ood_confs: scores de confiance obtenus avec react/ash
    """
    perturbations = []
    
    for conf in id_confs:
        perturbation = calc_perturbations(id_conf_sans, conf)
        perturbations.append(perturbation)
    
    poids = fct_ponderation(perturbations) #softmax, min_max_scaling
    confs_ponderees = [(a*b) for a,b in list(zip(ood_confs, poids))]
    
    return confs_ponderees
    
#####################################################################
# Pondération après avoir obtenu les perturbations
from abc import ABC

class Ponderator(ABC):
    """
    Ces classes donnent des scores finaux à partir d'une liste de scores bruts et de poids.
    Chacune va pondérer d'une manière différente
    """
    def __init__(self):
        super().__init__()
    def __call__(self, confs, weights):
        pass
    
class CoefficientPonderator(Ponderator) :
    def __init__(self, tau=1):
        """
        tau: le multiplicateur des poids
        """
        super().__init__()
        self.tau = tau
        
    def __call__(self, confs:np.ndarray, weights:np.ndarray) -> np.ndarray:   
        """
        args:
        confs (np.ndarray): (N, M), avec M nombre de modalités (2 si VF ou 3 si VFA)
        weights (np.ndarray): (M,), poids de chaque modalité
        
        returns:
        le score de confiance final
        """
        # le score de confiance final après pondération des modalités par leurs perturbations respectives (poids)
        ponderated_conf = np.sum(confs*self.tau*weights, axis=1).reshape(-1,1)
        return ponderated_conf

class NormalizationPonderator(Ponderator):
    def __init__(self):
        super().__init__()
    
    def __call__(self, confs, weights):    
        normalized_weights = weights / np.sum(weights)
        ponderated_conf = np.sum(confs*normalized_weights, axis=1).reshape(-1,1)
        return ponderated_conf