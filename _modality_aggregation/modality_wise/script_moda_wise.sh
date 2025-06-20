#!/bin/bash
#set -e  # Arrête le script si une commande échoue

cd "/data/maouche/MultiOOD/HMDB-rgb-flow"

# Save the evaluation files for HMDB 
# (to save evaluation files for ASH or ReAct, you should also run following line with options --use_ash or --use_react, same for other datasets):

# Using react
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --use_react --drop_modality 'video' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_hmdb_react_video.log | tee out_test_hmdb_react_video.log
echo "Saved evaluation files for HMDB ash\n"


#########
#cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
# # Save the evaluation files for UCF
# # Using react
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --drop_modality 'video' --ood_dataset 'UCF' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_ucf_react_video.log | tee out_test_react_video.log
echo "Saved evaluation files for UCF react\n"

# Evaluation for UCF
cd ..
# Using react
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_eval_ucf_react_video.log | tee out_eval_ucf_react_video.log
echo "Evaluation for UCF react finished\n"

####################

# cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
# # Save the evaluation files for HAC:

# # Using React
# python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --drop_modality 'video' --ood_dataset 'HAC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_test_hac_react_video.log | tee out_test_hac_react_video.log
# echo "Saved evaluation files for HAC react\n"

# # Evalauation for HAC
# cd ..
# # # Using react
# python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_react_video_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_eval_hac_react_video.log | tee out_eval_hac_react_video.log
# echo "Evaluation for HAC react finished\n"

# Save the evaluation files for EPIC:
#cd ../EPIC-rgb-flow/

# Using React
#python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log4_epic_react.txt | tee out_log4_epic_react.txt
#echo "Saved evaluation files for EPIC react\n"

# Evaluation for UCF 
# (change --postprocessor to different score functions, for VIM you should also pass --resume_file checkpoint.pt, where checkpoint.pt is the trained checkpoint, change --ood_dataset to UCF, EPIC, HAC, or Kinetics):
# cd ..

# # Using ash
# python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --use_ash --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log5.txt | tee out_log5.txt
# echo "Evaluation for UCF ash finished\n"

# # Using react
# python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --use_react --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log5.txt | tee out_log5.txt
# echo "Evaluation for UCF react finished\n