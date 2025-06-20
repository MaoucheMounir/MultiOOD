#!/bin/bash
#set -e  # Arrête le script si une commande échoue

cd "HMDB-rgb-flow/"

# Save the evaluation files for HMDB 
# (to save evaluation files for ASH or ReAct, you should also run following line with options --use_ash or --use_react, same for other datasets):
# Using ash
python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --use_ash --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log_hmdb_ash.txt | tee out_log_hmdb_ash.txt
echo "Saved evaluation files for HMDB ash\n"

# Using react
python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --use_react --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log_hmdb_react.txt | tee out_log_hmdb_react.txt
echo "Saved evaluation files for HMDB ash\n"




# Save the evaluation files for UCF 
# Using ash
python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_ash --ood_dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log2_ucf_ash.txt | tee out_log2_ucf_ash.txt
echo "Saved evaluation files for UCF ash \n"

# Using react
python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --ood_dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log2_ucf_react.txt | tee out_log2_ucf_react.txt
echo "Saved evaluation files for UCF react\n"

# Save the evaluation files for HAC:
# Using ash
python test_video_flow.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_ash --ood_dataset 'HAC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log3_hac_ash.txt | tee out_log3_hac_ash.txt
echo "Saved evaluation files for HAC ash\n"

# Using React
python test_video_flow.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --ood_dataset 'HAC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log3_hac_react.txt | tee out_log3_hac_react.txt
echo "Saved evaluation files for HAC react\n"



# Save the evaluation files for Kinetics (Indisponible, trop gros)
# python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'

# Save the evaluation files for EPIC:
cd ../EPIC-rgb-flow/

# Using ash (succes)
#python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_ash --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log4_epic_ash.txt | tee out_log4_epic_ash.txt
#echo "Saved evaluation files for EPIC ash\n"

# Using React
python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log4_epic_react.txt | tee out_log4_epic_react.txt
echo "Saved evaluation files for EPIC react\n"

# Evaluation for UCF 
# (change --postprocessor to different score functions, for VIM you should also pass --resume_file checkpoint.pt, where checkpoint.pt is the trained checkpoint, change --ood_dataset to UCF, EPIC, HAC, or Kinetics):
# cd ..

# # Using ash
# python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --use_ash --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log5.txt | tee out_log5.txt
# echo "Evaluation for UCF ash finished\n"

# # Using react
# python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --use_react --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log5.txt | tee out_log5.txt
# echo "Evaluation for UCF react finished\n"