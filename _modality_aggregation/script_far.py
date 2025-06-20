import os
from mounirood.config_mounir import FarOODFramework 

#-------------------------

framework = FarOODFramework("moda_wise") 
datasets, test_filename, _, modalities, \
hmdb_test_command_template, test_command_template, \
eval_command_template = framework.__dict__.values()

backbone = "baseline"
methods = ['msp'] #, ['msp', 'ebo', 'maxlogit', 'Mahalanobis', 'react', 'ash', 'gen', 'knn', 'vim']

sparsification_methods = {"ash_": "--use_ash"}

#-------------------------

with open("moda_wise_far_ash.sh", "w") as f:
    current_file = os.path.basename(__file__)
    f.write("# Généré par : "+current_file+"\n")
    
# Fichiers d'évaluation ID pour HMDB
    f.write('cd "/data/maouche/MultiOOD/HMDB-rgb-flow"\n')
    for modality in modalities:
        f.write(f"\n# {modality}\n")    
        for sparsification_suffix, sparsification_method in sparsification_methods.items():
            for method in methods:
                params = {
                    "sparsification_method": sparsification_method,
                    "backbone": backbone,
                    "modality": modality,
                    "sparsification_suffix": sparsification_suffix,
                }
                hmdb_test_command = hmdb_test_command_template.format(**params)            
                
                f.write(hmdb_test_command + "\n")
                f.write(f'echo "Saved evaluation files for HMDB {sparsification_suffix} {modality}"\n\n')
            
            
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
                        f.write('cd /data/maouche/MultiOOD/EPIC-rgb-flow\n')
                    else:
                        f.write('cd /data/maouche/MultiOOD/HMDB-rgb-flow\n')
                        
                    
                    f.write(test_command + "\n")
                    f.write(f'echo "Saved evaluation files for {dataset.label} {sparsification_suffix} {modality}"\n')
                    f.write('cd ..\n')
                    f.write(eval_command + "\n")
                    f.write(f'echo "Evaluation for {dataset.label} {sparsification_suffix} {modality} finished"\n\n')
