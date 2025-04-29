import os
import shutil
from tqdm import tqdm

working_dir = "/data/maouche/MultiOOD/UCF-101"
os.chdir(working_dir)

dest_dir = "/data/maouche/MultiOOD/UCF101/video/"
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

for root, dir, files in tqdm(os.walk(working_dir)):
    for filename in tqdm([file for file in files if file.endswith(".avi")]):
        
        src_path = os.path.join(working_dir, root)
        dest_path = dest_dir
        
        shutil.move(os.path.join(src_path, filename), os.path.join(dest_path, filename))

# for dir in os.listdir("."):   
#     if os.path.isdir(dir):
#         for file in os.listdir(dir):
#             if file.endswith(".wav"):
                