#!/bin/bash
#set -e  # Arrête le script si une commande échoue

cd "HMDB-rgb-flow/"

# Save the evaluation files for HMDB 
# (to save evaluation files for ASH or ReAct, you should also run following line with options --use_ash or --use_react, same for other datasets):
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_hmdb_baseline.txt | tee out_log_hmdb_baseline.txt
# echo "Saved evaluation files for HMDB\n"

# # Save the evaluation files for UCF: 
# python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'UCF' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_ucf_baseline.txt | tee out_log_ucf_baseline.txt
# echo "Saved evaluation files for UCF\n"

# # Save the evaluation files for HAC: 
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'HAC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_hac_baseline.txt | tee out_log_hac_baseline.txt
# echo "Saved evaluation files for HAC\n"

# # Save the evaluation files for Kinetics (Indisponible, trop gros)
# #python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'Kinetics' --appen 'baseline_best_' --resumef '/path/to/HMDB_far_ood_baseline.pt'

# # Save the evaluation files for EPIC:
# cd ../EPIC-rgb-flow/
# python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'EPIC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_epic_baseline.txt | tee out_log_epic_baseline.txt
# echo "Saved evaluation files for EPIC\n"


### Ash
# Save the evaluation files for HMDB 
# (to save evaluation files for ASH or ReAct, you should also run following line with options --use_ash or --use_react, same for other datasets):
#python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --use_ash --appen 'baseline_best_ash_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_hmdb_baseline_ash.txt | tee out_log_hmdb_baseline_ash.txt
#echo "Saved evaluation files for HMDB ash\n"

# Save the evaluation files for UCF: 
#python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_ash --ood_dataset 'UCF' --appen 'baseline_best_ash_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_ucf_baseline_ash.txt | tee out_log_ucf_baseline_ash.txt
#echo "Saved evaluation files for UCF ash\n"

# Save the evaluation files for HAC: 
#python test_video_flow.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_ash --ood_dataset 'HAC' --appen 'baseline_best_ash_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_hac_baseline_ash.txt | tee out_log_hac_baseline_ash.txt
#echo "Saved evaluation files for HAC ash\n"

# Save the evaluation files for EPIC:
#cd ../EPIC-rgb-flow/
#python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_ash --ood_dataset 'EPIC' --appen 'baseline_best_ash_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_epic_baseline_ash.txt | tee out_log_epic_baseline_ash.txt
#echo "Saved evaluation files for EPIC ash\n"


### React
# cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
# # Save the evaluation files for HMDB 
# # (to save evaluation files for ASH or ReAct, you should also run following line with options --use_ash or --use_react, same for other datasets):
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --use_react --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_hmdb_baseline_react.txt | tee out_log_hmdb_baseline_react.txt
# echo "Saved evaluation files for HMDB react\n"

# # Save the evaluation files for UCF: 
# python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --ood_dataset 'UCF' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_ucf_baseline_react.txt | tee out_log_ucf_baseline_react.txt
# echo "Saved evaluation files for UCF react\n"

# # Save the evaluation files for HAC: 
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --ood_dataset 'HAC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_hac_baseline_react.txt | tee out_log_hac_baseline_react.txt
# echo "Saved evaluation files for HAC react\n"

# Save the evaluation files for EPIC:
cd ../EPIC-rgb-flow/
python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --ood_dataset 'EPIC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_log_epic_baseline_react.txt | tee out_log_epic_baseline_react.txt
echo "Saved evaluation files for EPIC  react\n"
