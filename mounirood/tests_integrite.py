import numpy as np
from mounirood.framework import FrameworkFactory
from mounirood.tester import TesterVanilla
import subprocess
import sys, os
sys.path.append(os.path.abspath('/data/maouche/MultiOOD/HMDB-rgb-flow'))

def test_far_ood():
    ## ID
    
    # Papier
    args = {"appen": 'baseline_best_', "dataset":"HMDB", "ood_dataset": 'EPIC', "path": 'HMDB-rgb-flow/'}
    split = "test"
    conf_name = args["path"] + '/saved_files/id_'+args["dataset"]+'_conf_' + args["appen"] + split + '.npy'
    conf_papier = np.load(conf_name)

    # Moi
    framework = FrameworkFactory("far_ood")
    conf_moi = framework.get_confs("id", "none")

    # Test
    assert np.all(conf_papier.reshape(-1,1) == conf_moi.reshape(-1,1)), "Far OOD: Les données ID sont différentes"
    
    ## OOD
    
    for ood_dataset in ["EPIC", "HAC", 'UCF']:
        # Papier
        #ood_dataset = "EPIC"
        args = {"appen": 'baseline_best_', "dataset":"HMDB", "ood_dataset": ood_dataset, "path": 'HMDB-rgb-flow/'}
        split = "eval"
        conf_name = args["path"] + '/saved_files/id_'+args["dataset"]+'_ood_'+args["ood_dataset"]+'_conf_' + args["appen"] + split + '.npy'
        conf_papier = np.load(conf_name)

        # Moi
        framework = FrameworkFactory("far_ood")
        conf_moi = framework.get_vanilla_confs("ood", ood_dataset)

        # Test
        assert np.all(conf_papier.reshape(-1,1) == conf_moi.reshape(-1,1)), "Near OOD: Les données OOD sont différentes"
        
def test_near_ood():
    for dataset in ["HMDB", "UCF", "EPIC"]:
        
        # Papier
        args = {"appen": 'baseline_best_', "dataset":dataset, "path": 'HMDB-rgb-flow/'}
        split = "eval"
        conf_name = args["path"] + '/saved_files/id_'+args["dataset"]+'_near_ood_conf_' + args["appen"] + split + '.npy'
        conf_papier = np.load(conf_name)

        # Moi
        framework = FrameworkFactory("near_ood", dataset)
        conf_moi = framework.get_vanilla_confs("ood", dataset)
        # Test
        assert np.all(conf_papier.reshape(-1,1) == conf_moi.reshape(-1,1)), "Near OOD: Les données sont différents"
        
def tester_vfa():
    ## ID
    
    # Papier
    args = {"appen": 'baseline_best_', "ood_dataset": 'EPIC', "path": 'HMDB-rgb-flow/'}
    split = "test"
    conf_name = args["path"] + 'saved_files/id_'+args["ood_dataset"]+'_near_ood_conf_vfa_' + args["appen"] + split + '.npy'
    conf_papier = np.load(conf_name)

    # Moi
    framework = FrameworkFactory("vfa")
    conf_moi = framework.get_confs("id", "none")

    # Test
    assert np.all(conf_papier.reshape(-1,1) == conf_moi.reshape(-1,1)), "VFA: Les données ID sont différentes"
    
    ## OOD
    
    # Papier
    args = {"appen": 'baseline_best_', "ood_dataset": 'EPIC', "path": 'HMDB-rgb-flow/'}
    split = "eval"
    conf_name = args["path"] + 'saved_files/id_'+args["ood_dataset"]+'_near_ood_conf_vfa_' + args["appen"] + split + '.npy'
    conf_papier = np.load(conf_name)

    # Moi
    framework = FrameworkFactory("vfa")
    conf_moi = framework.get_confs("ood", "none")

    # Test
    assert np.all(conf_papier.reshape(-1,1) == conf_moi.reshape(-1,1)), "VFA: Les données OOD sont différentes "


def test_all():
    os.chdir("/data/maouche/MultiOOD")
    test_far_ood()
    test_near_ood()
    tester_vfa()
    print("OK")

if __name__== "main":
    os.chdir("/data/maouche/MultiOOD")
    test_far_ood()
    test_near_ood()
    tester_vfa()
    print("OK")

## Tests des résultats