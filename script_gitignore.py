import subprocess

with open('.gitignore', "r") as f:
    lines = f.readlines()

command_template = "git rm -r --cached "
    
for file in lines:
    commande = (command_template+file).split()
    
    print(commande)
    result = subprocess.run(commande, capture_output=True, text=True)

    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

