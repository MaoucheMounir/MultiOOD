# Faire l'agrégation mais en near ood
modalities = ["video", "flow", "audio"]
datasets = {"HMDB":("HMDB51", "HMDB-rgb-flow/HMDB_near_ood_baseline.pt"), 
            "UCF": ("UCF101", "HMDB-rgb-flow/UCF_near_ood_baseline.pt"),
            "EPIC": ("EPIC-KITCHENS", "EPIC-rgb-flow/EPIC_near_ood_baseline.pt")}
sparsification_methods = {"react_": "--use_react"}


test_command_template = (
    "python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/{datapath}/ "
    "--bsz 16 --num_workers 2 --near_ood --dataset '{dataset}' {sparsification_method} "
    "--drop_modality {modality} --appen 'baseline_best_' "
    "--resumef '/data/maouche/MultiOOD/{resume_path}' "
    "2>error_test_{dataset}_{sparsification_suffix}{modality}.log | "
    "tee out_test_{dataset}_{sparsification_suffix}{modality}.log"
)


eval_command_template = (
    "python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_{sparsification_suffix}{modality}_' "
    "--dataset '{dataset}' --path 'HMDB-rgb-flow/' "
    "2>error_eval_{dataset}_{sparsification_suffix}{modality}.log | "
    "tee out_eval_{dataset}_{sparsification_suffix}{modality}.log"
)

with open("script_all_react_moda_wise_near_ood.sh", "w") as f:

# Fichiers d'évaluation 
    for modality in modalities:
        for sparsification_suffix, sparsification_method in sparsification_methods.items(): 
            for dataset, (dataset_path, dataset_cpt) in datasets.items():    
                params = {
                    "datapath": dataset_path, 
                    "sparsification_method": sparsification_method,
                    "modality": modality,
                    "sparsification_suffix": sparsification_suffix,
                    "dataset": dataset,
                    "resume_path": dataset_cpt,
                      
                }

                test_command = test_command_template.format(**params)
                if dataset == "EPIC":
                    test_command = test_command.replace("test_video_flow_moda_wise.py", "test_video_flow_epic_moda_wise.py")
                    test_command = test_command.replace('--near_ood ', "")
                
                eval_command = eval_command_template.format(**params)
                
                
                if dataset != "EPIC":
                    f.write('cd /data/maouche/MultiOOD/HMDB-rgb-flow\n')
                else:
                    f.write('cd /data/maouche/MultiOOD/EPIC-rgb-flow\n')
                if dataset == "EPIC":
                    f.write(test_command + "\n")
                    f.write(f'echo "Saved evaluation files for {dataset} {sparsification_suffix} {modality}"\n')
                f.write('cd ..\n')
                f.write(eval_command + "\n")
                f.write(f'echo "Evaluation for {dataset} {sparsification_suffix} {modality} finished"\n\n')
