import os
import shutil

nom_errone = "v_HandStandPushups"
nom_correct = "v_HandstandPushups"
path = "/data/maouche/MultiOOD/UCF101/video"

with open("/data/maouche/MultiOOD/HMDB-rgb-flow/splits/UCF_test_near_ood.txt", "r") as f:
    read_names = f.read()
cpt = 0
for file in os.listdir(path):
    if nom_errone in file:
        file_corrige = file.replace(nom_errone, nom_correct)
        shutil.move(os.path.join(path,file), os.path.join(path,file_corrige) )
