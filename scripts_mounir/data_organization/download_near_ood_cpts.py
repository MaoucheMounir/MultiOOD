import subprocess

cpt_links = [
    "HMDB_near_ood_a2d_npmix",
    "HMDB_near_ood_baseline",
    "EPIC_near_ood_a2d_npmix",
    "EPIC_near_ood_baseline",
    "UCF_near_ood_a2d_npmix",
    "UCF_near_ood_baseline",
]

link_template = "https://huggingface.co/datasets/hdong51/MultiOOD/resolve/main/checkpoints/{}.pt?download=true"

for cpt in cpt_links:
    commande = "wget "+link_template.format(cpt)
    print(commande)
    commande = commande.split()
    result = subprocess.run(commande, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)