import os
import shutil

working_dir = "/data/maouche/MultiOOD/EPIC-KITCHENS/rgb/train/D3"

os.chdir(working_dir)

for dir in os.listdir("."):
    if os.path.isdir(dir):
        for file in os.listdir(dir):
            if file.endswith(".wav"):
                shutil.move(os.path.join(working_dir, dir, file), os.path.join(working_dir, file))