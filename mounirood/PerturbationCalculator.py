import numpy as np
from typing import Union

class PerturbationCalculator:
    def __init__(self, method:str="mse"):
        self.method = method
        self._check_method()

    def _check_method(self):
        valid_methods = ["mse", "diff", "diff_vector", "diff_vector_abs"]
        if self.method not in valid_methods:
            raise ValueError(f"Méthode non reconnue: {self.method}. Choisir parmi {valid_methods}")

    def __call__(self, conf1: Union[np.ndarray, list], conf2: Union[np.ndarray, list]) -> np.ndarray:
        """
        conf1 est en général le score de base et conf2 avec react
        """
        
        conf1 = np.array(conf1)
        conf2 = np.array(conf2)

        if self.method == "mse":
            return np.mean((conf1 - conf2) ** 2)
        elif self.method == "diff":
            return np.mean(conf1 - conf2)
        elif self.method == "diff_vector":
            return conf2 - conf1
        elif self.method == "diff_vector_abs":
            return np.abs(conf2 - conf1)
