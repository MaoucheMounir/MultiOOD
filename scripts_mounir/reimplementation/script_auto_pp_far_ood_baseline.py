import subprocess

methods = ['msp', 'ebo', 'maxlogit', 'Mahalanobis', 'react', 'ash', 'gen', 'knn', 'vim']
datasets = ["UCF", "EPIC", "HAC"]
layer_proc = ["", "ash_", "react_"]
backbones = ["baseline"]#, "a2d_npmix"]
checkpoint_path = " --resume_file /data/maouche/MultiOOD/{}-rgb-flow/{}_near_ood_{}.pt"
#command_template = "python eval_video_flow_far_ood.py --postprocessor {method} --appen '{backbone}_best_{layer_proc}' --dataset 'HMDB' --ood_dataset '{dataset}' --path 'HMDB-rgb-flow/'{cpt} 2>logs_eval_far/error_log_pp_{method}_{backbone}_{layer_proc}{dataset}.txt | tee logs_eval_far/out_log_pp_{method}_{backbone}_{layer_proc}{dataset}.txt"
command_template = "python eval_video_flow_far_ood.py --postprocessor {} --appen '{}_best_{}' --dataset 'HMDB' --ood_dataset '{}' --path 'HMDB-rgb-flow/'{} 2>logs_eval_far/error_log_pp_{}_{}_{}{}.txt | tee logs_eval_far/out_log_pp_{}_{}_{}{}.txt"
#command_template = "python eval_video_flow_far_ood.py --postprocessor {} --appen 'a2d_npmix_best_{}' --dataset 'HMDB' --ood_dataset '{}' --path 'HMDB-rgb-flow/'{} 2>error_log_pp_{}_{}{}.txt | tee out_log_pp_{}_{}{}.txt"da

#hmdb
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --path 'HMDB-rgb-flow/'
#ucf
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'UCF' --path 'HMDB-rgb-flow/'
#epic
#python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'EPIC' --path 'EPIC-rgb-flow/'
# method, backbone, layer_proc, dataset, cpt, ...
with open("script_auto_pp_far_ood_baseline.sh", "w") as f:
    for backbone in backbones: #baseline ou ++
        for method in methods:
            for dataset in datasets:
                if method == "vim":
                    ds = "EPIC" if dataset == "EPIC" else "HMDB"
                    cpt = checkpoint_path.format(ds, dataset, backbone)
                else:
                    cpt = ""
                for proc in layer_proc:
                    if ( (proc in ["ash_", "react_"]) and (method not in ["ash", "react"]) ) \
                    or ( (method in ["ash", "react"]) and (method != proc.strip("_")) ):#proc not in ["ash_", "react_"]) ):
                        continue
                    
                    l = (method, backbone, proc, dataset)
                    commande = command_template.format(*l, cpt, *(l*2))
                    #commande = command_template.format(method, backbone, proc, ood_dataset, cpt, method, proc, ood_dataset, method, proc, ood_dataset)
                    #print(commande)
                    #commande = commande.split()
                    #result = subprocess.run(commande, capture_output=True, text=True)
                    #print("stdout:", result.stdout)
                    #print("stderr:", result.stderr)
                    #print(f"Evaluation for {dataset} {method} {proc} finished")
                    f.write(commande+"\n")
                    f.write(f'echo "Evaluation for {backbone} {method} {dataset} {proc} finished"\n')