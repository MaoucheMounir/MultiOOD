#!/bin/bash
#set -e  # Arrête le script si une commande échoue

### Evaluation for Mahalanobis  
# (change --postprocessor to different score functions, for VIM you should also pass --resume_file checkpoint.pt, where checkpoint.pt is the trained checkpoint, change --ood_dataset to UCF, EPIC, HAC, or Kinetics):

## Vanilla

# UCF
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_ucf.txt | tee out_log_pp_maha_ucf.txt
echo "Evaluation for UCF Mahalanobis finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_hac.txt | tee out_log_pp_maha_hac.txt
echo "Evaluation for HAC Mahalanobis finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_epic.txt | tee out_log_pp_maha_epic.txt
echo "Evaluation for EPIC Mahalanobis finished"


### MaxLogit
# UCF
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_ucf.txt | tee out_log_pp_maxlog_ucf.txt
echo "Evaluation for UCF MaxLogit finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_hac.txt | tee out_log_pp_maxlog_hac.txt
echo "Evaluation for HAC MaxLogit finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_epic.txt | tee out_log_pp_maxlog_epic.txt
echo "Evaluation for EPIC MaxLogit finished"

### MSP
# UCF
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_ucf.txt | tee out_log_pp_msp_ucf.txt
echo "Evaluation for UCF MSP finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_hac.txt | tee out_log_pp_msp_hac.txt
echo "Evaluation for HAC MSP finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_epic.txt | tee out_log_pp_msp_epic.txt
echo "Evaluation for EPIC MSP finished"

### gen
# UCF
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_ucf.txt | tee out_log_pp_gen_ucf.txt
echo "Evaluation for UCF GEN finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_hac.txt | tee out_log_pp_gen_hac.txt
echo "Evaluation for HAC GEN finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_epic.txt | tee out_log_pp_gen_epic.txt
echo "Evaluation for EPIC GEN finished"

### knn
# UCF
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_ucf.txt | tee out_log_pp_knn_ucf.txt
echo "Evaluation for UCF KNN finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_hac.txt | tee out_log_pp_knn_hac.txt
echo "Evaluation for HAC KNN finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_epic.txt | tee out_log_pp_knn_epic.txt
echo "Evaluation for EPIC KNN finished"
