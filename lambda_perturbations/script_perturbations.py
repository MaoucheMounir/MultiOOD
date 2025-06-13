import subprocess

frameworks = ["far_ood", "near_ood", "vfa" ]
layer_procs = ["react", "ash"]
datasets = ["HMDB", "UCF", "EPIC"]

command_template = "python -m lambda_perturbations.lambda_perturbations --framework {framework} --layer_proc {layer_proc}"

for framework in frameworks:
    for layer_proc in layer_procs:
        if framework == "near_ood":
            for dataset in datasets:
                commande = command_template.format(framework=framework, layer_proc=layer_proc)+f" --dataset {dataset}"
                print(commande)
                commande = commande.split()
                result = subprocess.run(commande, capture_output=True, text=True)
                print("stdout:", result.stdout)
                print("stderr:", result.stderr)
        else:
            commande = command_template.format(framework=framework, layer_proc=layer_proc)
            print(commande)
            commande = commande.split()
            result = subprocess.run(commande, capture_output=True, text=True)
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)