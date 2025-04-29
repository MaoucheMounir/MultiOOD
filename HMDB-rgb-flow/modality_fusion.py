from mmaction.apis import init_recognizer
import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from dataloader_video_flow import EPICDOMAIN
from dataloader_video_flow_hac import HACDOMAIN

rgb_model_path = "/data/maouche/MultiOOD/HMDB-rgb-flow/pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth"
flow_model_path = "/data/maouche/MultiOOD/HMDB-rgb-flow/pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth"
mode_path = "/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt"

rgb_model = torch.load(rgb_model_path)

###################################################################
class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
  
    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)
    
mlp_cls = Encoder(input_dim=v_dim+f_dim, out_dim=num_class)
mlp_cls = mlp_cls.cuda()

def validate_one_step(model, clip, labels, flow, model_flow):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)  # 16,1024,8,14,14
        v_feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)

        f_feat = model_flow.module.backbone.get_feature(flow)  # 16,1024,8,14,14
        f_feat = model_flow.module.backbone.get_predict(f_feat)
        f_predict, f_emd = model_flow.module.cls_head(f_feat)

        if args.use_ash:
            v_emd = ash_b(v_emd.view(v_emd.size(0), -1, 1, 1))
            v_emd = v_emd.view(v_emd.size(0), -1)
            f_emd = ash_b(f_emd.view(f_emd.size(0), -1, 1, 1))
            f_emd = f_emd.view(f_emd.size(0), -1)

        if args.use_react:
            v_emd = v_emd.clip(max=args.v_thr)
            v_emd = v_emd.view(v_emd.size(0), -1)
            f_emd = f_emd.clip(max=args.f_thr)
            f_emd = f_emd.view(f_emd.size(0), -1)

        predict = mlp_cls(v_emd, f_emd)
        feature = torch.cat((v_emd, f_emd), dim=1)

    return predict, feature, v_predict, f_predict
###############################################################

# Commencer par RGB seul
## Charger les données
## Appliquer le modèle et récupérer les features 
## React/ash ; mahalanobis/cosinus
### Mahalanobis/Cosinus: 
#   Récupérer les embeddings de chaque modalité à une certaine couche 
#   (doit être un vecteur)
#   Faire la distance entre les deux
args = None
cfg, cfg_flow = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement des données
if args.far_ood:
    if args.ood_dataset == "HAC":
        eval_dataset = HACDOMAIN(cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=False,
                                                    pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'eval': eval_dataloader}
    splits = ['eval']

# Chargement du modèle


# Application du modèle
for split in splits:
        print(split)
        pred_list, conf_list, label_list, output_list, feature_list = [], [], [], [], []
        for clip, spectrogram, labels in tqdm(dataloaders[split]):
            output, feature, output_v, output_f = validate_one_step(model, clip, labels, spectrogram, model_flow)
            score = torch.softmax(output, dim=1)
            conf, pred = torch.max(score, dim=1)
            output_list.append(output.cpu())
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(labels.cpu())
            feature_list.append(feature.cpu())

        output_list = torch.cat(output_list).numpy()
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        feature_list = torch.cat(feature_list).numpy()


# A vérifier mais on dirait que dans leur variable "model" il y a deux cas,
# le premier où il charge le checkpoint, et un deuxième où elle est composée du backbone/checkpoint + le classifieur
# adapter validate_one_step