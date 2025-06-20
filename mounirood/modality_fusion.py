import numpy as np

def calc_perturbations(conf1:np.ndarray, conf2:np.ndarray, method="mse") -> np.ndarray:
    """
    args:
    conf1, conf2 (np.ndarray, list)
    """
    
    if method == "mse":
        return np.mean((conf1 - conf2) ** 2)
    if method == "diff":
        return np.mean(conf1-conf2)

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
    
    for i, conf in enumerate(id_confs):
        perturbation = calc_perturbations(id_conf_sans, conf)
        perturbations.append(perturbation)
    
    poids = fct_ponderation(perturbations) #softmax, min_max_scaling
    confs_ponderees = [(a*b) for a,b in list(zip(ood_confs, poids))]
    
    return confs_ponderees
    
# def ponderation_predictor(confs:np.ndarray, weights:np.ndarray, tau:int=1) -> np.ndarray:
#     """
#     args:
#     confs (np.ndarray): (N, M), avec M nombre de modalités (2 si VF ou 3 si VFA)
#     weights (np.ndarray): (M,), poids de chaque modalité
    
#     returns:
#     le score de confiance final et la prédiction éventuellement 
#     """
#     # le score de confiance final après pondération des modalités par leurs perturbations respectives (poids)
#     ponderated_conf = np.sum(confs*tau*weights, axis=1).reshape(-1,1)
#     return ponderated_conf

class ponderation_predictor() :
    def __init__(self, tau):
        """
        tau: le multiplicateur des poids
        """
        self.tau = tau
    def __call__(self, confs:np.ndarray, weights:np.ndarray, tau:float=1) -> np.ndarray:   
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

def normalization_predictor(confs, weights):
    normalized_weights = weights / np.sum(weights)
    ponderated_conf = np.sum(confs*normalized_weights, axis=1).reshape(-1,1)
    return ponderated_conf