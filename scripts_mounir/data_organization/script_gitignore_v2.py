import subprocess

# Extensions à ignorer
extensions = ['.log']
ext = ".log"
# Liste les fichiers suivis par Git
result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
tracked_files = result.stdout.splitlines()

for file in tracked_files:
    #if any(file.endswith(ext) for ext in extensions):
    if file.endswith(ext):
        print(f"Removing {file} from Git tracking...")
        subprocess.run(['git', 'rm', '--cached ', file])
