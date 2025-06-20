import subprocess

methods = ['msp', 'ebo', 'maxlogit', 'Mahalanobis', 'ash', 'react', 'knn', 'gen', 'vim']
ood_datasets = ["UCF", "EPIC", "HAC"]
layer_proc = ["", "ash_", "react_"]
checkpoint_path = " --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt"
command_template = "python eval_video_flow_far_ood.py --postprocessor {} --appen 'a2d_npmix_best_{}' --dataset 'HMDB' --ood_dataset '{}' --path 'HMDB-rgb-flow/'{} 2>error_log_pp_{}_{}{}.txt | tee out_log_pp_{}_{}{}.txt"

with open("test_file.txt", "w") as f:

    for method in methods:
        if method == "vim":
            cpt = checkpoint_path
        else:
            cpt = ""
                
        for ood_dataset in ood_datasets:
            for proc in layer_proc:
                commande = command_template.format(method, proc, ood_dataset, cpt, method, proc, ood_dataset, method, proc, ood_dataset)
                print(commande)
                commande = commande.split()
                result = subprocess.run(commande, capture_output=True, text=True)
                print("stdout:", result.stdout)
                print("stderr:", result.stderr)
                print(f"Evaluation for {ood_dataset} {method} {proc} finished")
                # f.write(commande+"\n")
                # f.write(f"Evaluation for {ood_dataset} {method} {proc} finished\n")