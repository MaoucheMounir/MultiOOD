import os
import subprocess
from mounirood.framework import FrameworkFactory 

#-------------------------
def exec_command(commande:str, message:str=""):
    print(commande + "\n")
    print(message)
    commande = commande.split()
    result = subprocess.run(commande, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

framework = FrameworkFactory("far_ood") 
datasets, test_filename, _, modalities, \
hmdb_test_command_template, test_command_template, \
eval_command_template = framework.__dict__.values()

backbone = "baseline"
methods = ['msp'] #, ['msp', 'ebo', 'maxlogit', 'Mahalanobis', 'react', 'ash', 'gen', 'knn', 'vim']

sparsification_methods = {"react_":"--use_react", "ash_": "--use_ash"}

#-------------------------

#f.write("# Généré par : "+current_file+"\n")
    
# Fichiers d'évaluation ID pour HMDB
#    f.write('cd "/data/maouche/MultiOOD/HMDB-rgb-flow"\n')
for modality in modalities:
    print(f"\n# {modality}\n")    
    for sparsification_suffix, sparsification_method in sparsification_methods.items():
        for method in methods:
            params = {
                "sparsification_method": sparsification_method,
                "backbone": backbone,
                "modality": modality,
                "sparsification_suffix": sparsification_suffix,
            }
            hmdb_test_command = hmdb_test_command_template.format(**params)            
            exec_command(hmdb_test_command, f"Saved evaluation files for HMDB {sparsification_suffix} {modality}\n\n")
            
                            
# Fichiers d'évaluation OOD
    for modality in modalities:
        for sparsification_suffix, sparsification_method in sparsification_methods.items(): 
            for dataset in datasets:    
                if dataset.label == "EPIC" and (not test_filename.endswith("epic")): #dataset.label.endswith("epic.py"):
                    test_filename += "_epic"
                elif dataset.label != "EPIC":
                    test_filename = test_filename.replace("_epic", "")
                    
                for method in methods:
                    params = {
                        "test_filename": test_filename,
                        "datapath": dataset.name, 
                        "sparsification_method": sparsification_method,
                        "modality": modality,
                        "sparsification_suffix": sparsification_suffix,
                        "dataset": dataset.label,
                        "backbone": backbone,
                        "resume_path": dataset.backbone_path(backbone),
                        "postprocessor": method
                    }
                    
                    test_command = test_command_template.format(**params)
                    eval_command = eval_command_template.format(**params)
                    
                    if dataset.label == "EPIC":
                        os.chdir('/data/maouche/MultiOOD/EPIC-rgb-flow')
                    else:
                        os.chdir('/data/maouche/MultiOOD/HMDB-rgb-flow')
                    
                    exec_command(test_command, f"Saved evaluation files for {dataset.label} {sparsification_suffix} {modality}\n\n")
                    exec_command(test_command, f"Evaluated {dataset.label} {sparsification_suffix} {modality}\n\n")
