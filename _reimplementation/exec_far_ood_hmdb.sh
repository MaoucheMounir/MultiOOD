#!/bin/bash
#set -e  # Arrête le script si une commande échoue

cd "HMDB-rgb-flow/"

# Save the evaluation files for HMDB 
# (to save evaluation files for ASH or ReAct, you should also run following line with options --use_ash or --use_react, same for other datasets):
#python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log.txt | tee out_log.txt
#echo "Saved evaluation files for HMDB\n"

# Save the evaluation files for UCF (Succes)
#python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log2.txt | tee out_log2.txt
#echo "Saved evaluation files for UCF\n"

# Save the evaluation files for HAC: (Succes)
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'HAC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log3.txt | tee out_log3.txt
# echo "Saved evaluation files for HAC\n"

# Save the evaluation files for Kinetics (Indisponible, trop gros)
# python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'

# Save the evaluation files for EPIC:
cd ../EPIC-rgb-flow/
python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log4.txt | tee out_log4.txt
echo "Saved evaluation files for EPIC\n"

# Evaluation for UCF 
# (change --postprocessor to different score functions, for VIM you should also pass --resume_file checkpoint.pt, where checkpoint.pt is the trained checkpoint, change --ood_dataset to UCF, EPIC, HAC, or Kinetics):

# Succes
#cd ..
#python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log5.txt | tee out_log5.txt
#echo "Evaluation for UCF finished\n"