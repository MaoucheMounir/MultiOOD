# change the methods and the datasets
#datasets = ["HMDB", "UCF", "EPIC"]
# HMDB
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --path 'HMDB-rgb-flow/'
#UCF
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'UCF' --path 'HMDB-rgb-flow/'

# EPIC (Déplacer les savefiles de epic vers le dossier hmdb avec les autres)
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'EPIC' --path 'HMDB-rgb-flow/'