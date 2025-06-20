# A la base pour faire du near_ood_moda_wise
# Faire l'agrégation mais en near ood (VF)
from mounirood.config_mounir import NearOODFramework 

#-------------------------

framework = NearOODFramework("vfa", "moda_wise") 
datasets, test_filename, _, modalities, \
test_command_template, eval_command_template = framework.__dict__.values()

backbone = "baseline"
methods = ['msp'] #, ['msp', 'ebo', 'maxlogit', 'Mahalanobis', 'react', 'ash', 'gen', 'knn', 'vim']

sparsification_methods = {"react_": "--use_react"}

#-------------------------



with open("test_new_script_near.sh", "w") as f:

# Fichiers d'évaluation 
    for modality in modalities:
        for sparsification_suffix, sparsification_method in sparsification_methods.items(): 
            for dataset in datasets:    
                if dataset.label == "EPIC" and (not test_filename.endswith("epic")): #dataset.label.endswith("epic.py"):
                    test_filename += "_epic"
                elif dataset.label != "EPIC":
                    test_filename = test_filename.replace("_epic", "")
                    
                for method in methods:
                    params = {
                        "test_filename": test_filename+"_epic" if dataset == "EPIC" else test_filename,
                        "datapath": dataset.name, 
                        "sparsification_method": sparsification_method,
                        "modality": modality,
                        "sparsification_suffix": sparsification_suffix,
                        "dataset": dataset.label,
                        "backbone": backbone,
                        "resume_path": dataset.backbone_path(backbone),
                        "postprocessor": method
                    }

                    if dataset.label == "EPIC":
                        test_command = test_command.replace('--near_ood ', "")
                    
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
