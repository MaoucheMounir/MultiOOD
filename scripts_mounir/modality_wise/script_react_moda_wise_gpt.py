modalities = ["video", "flow"]
datasets = {"UCF": "UCF101", "HAC": "HAC", "EPIC": "EPIC-KITCHENS"}
sparsification_methods = {"react_": "--use_react"}

hmdb_test_command_template = (
    "python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ "
    "--bsz 16 --num_workers 2 --dataset 'HMDB' {sparsification_method} "
    "--drop_modality {modality} --appen 'baseline_best_' "
    "--resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' "
    "2>error_test_HMDB_{sparsification_suffix}{modality}.log | "
    "tee out_test_HMDB_{sparsification_suffix}{modality}.log"
)

test_command_template = (
    "python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/{datapath}/ "
    "--bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' {sparsification_method} "
    "--drop_modality {modality} --ood_dataset '{ood_dataset_name}' --appen 'baseline_best_' "
    "--resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' "
    "2>error_test_{ood_dataset_name}_{sparsification_suffix}{modality}.log | "
    "tee out_test_{ood_dataset_name}_{sparsification_suffix}{modality}.log"
)

eval_command_template = (
    "python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_{sparsification_suffix}{modality}_' "
    "--dataset 'HMDB' --ood_dataset '{ood_dataset_name}' --path 'HMDB-rgb-flow/' "
    "2>error_eval_{ood_dataset_name}_{sparsification_suffix}{modality}.log | "
    "tee out_eval_{ood_dataset_name}_{sparsification_suffix}{modality}.log"
)
with open("script_all_react_moda_wise.sh", "w") as f:
    # Fichiers d'évaluation ID pour HMDB
    f.write('cd "/data/maouche/MultiOOD/HMDB-rgb-flow"\n')
    for modality in modalities:
        for sparsification_suffix, sparsification_method in sparsification_methods.items():
            params = {
                "datapath": "HMDB51", 
                "sparsification_method": sparsification_method,
                "modality": modality,
                "sparsification_suffix": sparsification_suffix,
            }
            hmdb_test_command = hmdb_test_command_template.format(**params)            
            
            f.write(hmdb_test_command + "\n")
            f.write(f'echo "Saved evaluation files for HMDB {sparsification_suffix} {modality}"\n\n')
            
            
# Fichiers d'évaluation OOD pour UCF, HAC, EPIC
    for modality in modalities:
        for sparsification_suffix, sparsification_method in sparsification_methods.items(): 
            for ood_dataset_name, ood_dataset_path in datasets.items():    
                params = {
                    "datapath": ood_dataset_path, 
                    "sparsification_method": sparsification_method,
                    "modality": modality,
                    "sparsification_suffix": sparsification_suffix,
                    "ood_dataset_name": ood_dataset_name  
                }

                test_command = test_command_template.format(**params)
                if ood_dataset_name == "EPIC":
                    test_command = test_command.replace("test_video_flow_moda_wise.py", "test_video_flow_epic_moda_wise.py")
                
                eval_command = eval_command_template.format(**params)
                
                if ood_dataset_name != "EPIC":
                    f.write('cd "/data/maouche/MultiOOD/HMDB-rgb-flow"\n')
                else:
                    f.write('cd "/data/maouche/MultiOOD/EPIC-rgb-flow"\n')
                f.write(test_command + "\n")
                f.write(f'echo "Saved evaluation files for {ood_dataset_name} {sparsification_suffix} {modality}"\n')
                f.write('cd ..\n')
                f.write(eval_command + "\n")
                f.write(f'echo "Evaluation for {ood_dataset_name} {sparsification_suffix} {modality} finished"\n\n')
