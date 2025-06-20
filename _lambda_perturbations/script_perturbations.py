import subprocess

framework_names = ["far_ood", "near_ood", "vfa" ]
layer_procs = ["react", "ash"]
datasets = ["HMDB", "UCF", "EPIC"]

command_template = "python -m _lambda_perturbations.lambda_perturbations --framework {framework_name} --layer_proc {layer_proc} --save_results"

for framework_name in framework_names:
    for layer_proc in layer_procs:
        if framework_name == "near_ood":
            for dataset in datasets:
                commande = command_template.format(framework_name=framework_name, layer_proc=layer_proc)+f" --dataset {dataset}"
                print(commande)
                commande = commande.split()
                result = subprocess.run(commande, capture_output=True, text=True)
                print("stdout:", result.stdout)
                print("stderr:", result.stderr)
        else:
            commande = command_template.format(framework_name=framework_name, layer_proc=layer_proc)
            print(commande)
            commande = commande.split()
            result = subprocess.run(commande, capture_output=True, text=True)
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)