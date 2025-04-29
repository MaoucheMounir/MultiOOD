cd HMDB-rgb-flow/
### A2D+NP-MIX
## none
# HMDB
#python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_a2d_npmix.pt' 2>error_log_hmdb_a2dnpm_.txt | tee out_log_hmdb_a2dnpm_.txt
#echo 'Near OOD test for HMDB _a2dnpm_ done'

# UCF
# python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_a2d_npmix.pt' 2>error_log_ucf_a2dnpm_.txt | tee out_log_ucf_a2dnpm_.txt
# echo 'Near OOD test for UCF _a2dnpm_ done'

## ash
# HMDB
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_ash --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_a2d_npmix.pt' 2>error_log_hmdb_a2dnpm_ash.txt | tee out_log_hmdb_a2dnpm_ash.txt
# echo 'Near OOD test for HMDB _a2dnpm_ ash done'

# UCF
# python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_ash --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_a2d_npmix.pt' 2>error_log_ucf_a2dnpm_ash.txt | tee out_log_ucf_a2dnpm_ash.txt
# echo 'Near OOD test for UCF _a2dnpm_ ash done'

## react
# HMDB
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_react --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_a2d_npmix.pt' 2>error_log_hmdb_a2dnpm_react.txt | tee out_log_hmdb_a2dnpm_react.txt
# echo 'Near OOD test for HMDB _a2dnpm_ react done'

# UCF
# python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_react --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_a2d_npmix.pt' 2>error_log_ucf_a2dnpm_react.txt | tee out_log_ucf_a2dnpm_react.txt
# echo 'Near OOD test for UCF _a2dnpm_ react done'

######################################################################################################################################################################################################################################################################

### Baseline
## none
# HMDB
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_baseline.pt' 2>error_log_hmdb_baseline.txt | tee out_log_hmdb_baseline.txt
# echo 'Near OOD test for HMDB done'

# UCF
# python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_baseline.pt' 2>error_log_ucf_baseline.txt | tee out_log_ucf_baseline.txt
# echo 'Near OOD test for UCF done'

## ash
# HMDB
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_ash --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_baseline.pt' 2>error_log_hmdb_baseline_ash.txt | tee out_log_hmdb_baseline_ash.txt
# echo 'Near OOD test for HMDB ash done'

# UCF
# python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_ash --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_baseline.pt' 2>error_log_ucf_baseline_ash.txt | tee out_log_ucf_baseline_ash.txt
# echo 'Near OOD test for UCF ash done'

## react
# HMDB
# python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_react --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_baseline.pt' 2>error_log_hmdb_baseline_react.txt | tee out_log_hmdb_baseline_react.txt
# echo 'Near OOD test for HMDB done'

# UCF
# python test_video_flow.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_react --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_baseline.pt' 2>error_log_ucf_baseline_react.txt | tee out_log_ucf_baseline_react.txt
# echo 'Near OOD test for UCF done'

###############################################################################################################################################################################################################################################################################################

# EPIC 
cd ../EPIC-rgb-flow/

## none
### A2D+NP-MIX
# python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_a2d_npmix.pt' 2>error_log_epic_a2dnpm_.txt | tee out_log_epic_a2dnpm_.txt
# echo 'Near OOD test for EPIC _a2dnpm_ done'

### Baseline
# python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_baseline.pt' 2>error_log_epic_baseline.txt | tee out_log_epic_baseline.txt
# echo 'Near OOD test for EPIC done'

## ash
### A2D+NP-MIX
# python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_ash --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_a2d_npmix.pt' 2>error_log_epic_a2dnpm_ash.txt | tee out_log_epic_a2dnpm_ash.txt
# echo 'Near OOD test for EPIC _a2dnpm_ ash done'

### Baseline
# python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_ash --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_baseline.pt' 2>error_log_epic_baseline_ash.txt | tee out_log_epic_baseline_ash.txt
# echo 'Near OOD test for EPIC ash done'

## react
### A2D+NP-MIX
python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_a2d_npmix.pt' 2>error_log_epic_a2dnpm_react.txt | tee out_log_epic_a2dnpm_react.txt
echo 'Near OOD test for EPIC _a2dnpm_ done'

### Baseline
python test_video_flow_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_baseline.pt' 2>error_log_epic_baseline_react.txt | tee out_log_epic_baseline_react.txt
echo 'Near OOD test for EPIC done'

cd ..