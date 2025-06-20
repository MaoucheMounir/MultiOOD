import subprocess
import argparse

ood_datasets = {"UCF": "/data/maouche/MultiOOD/UCF101/",
    "HAC": "/data/maouche/MultiOOD/HAC/",
    "EPIC": "/data/maouche/MultiOOD/EPIC-KITCHENS/"} # le chemin est à mettre dans l'argument --datapath

backbone = {"baseline": "/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt",
    "a2d_npmix": "/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt"}

data_path = {} 

command_template = "python test_video_flow.py --datapath {} --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'UCF' --appen '{}_best_' --resumef {} 2>error_log_{}.txt | tee out_log_{}.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, default='baseline') # "baseline", "a2d_npmix"


with open("test_file2.sh", "w") as f:
    for dataset in ood_datasets:
        commande = command_template.format(method, proc, ood_dataset, cpt, method, proc, ood_dataset, method, proc, ood_dataset)
        print(commande)
        commande = commande.split()
        result = subprocess.run(commande, capture_output=True, text=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        print(f"Evaluation for {ood_dataset} {method} {proc} finished")
        # f.write(commande+"\n")
        # f.write(f"Evaluation for {ood_dataset} {method} {proc} finished\n")