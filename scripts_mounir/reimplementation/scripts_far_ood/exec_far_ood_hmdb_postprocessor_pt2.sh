#!/bin/bash
#set -e  # Arrête le script si une commande échoue
  
# (change --postprocessor to different score functions, for VIM you should also pass --resume_file checkpoint.pt, where checkpoint.pt is the trained checkpoint, change --ood_dataset to UCF, EPIC, HAC, or Kinetics):

### ash

# UCF
python eval_video_flow_far_ood.py --postprocessor ash --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_ash_ucf.txt | tee out_log_pp_ash_ucf.txt
echo "Evaluation for UCF ash finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor ash --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_ash_hac.txt | tee out_log_pp_ash_hac.txt
echo "Evaluation for HAC ash finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor ash --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_ash_epic.txt | tee out_log_pp_ash_epic.txt
echo "Evaluation for EPIC ash finished"

### react

# UCF
python eval_video_flow_far_ood.py --postprocessor react --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_react_ucf.txt | tee out_log_pp_react_ucf.txt
echo "Evaluation for UCF react finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor react --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_react_hac.txt | tee out_log_pp_react_hac.txt
echo "Evaluation for HAC react finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor react --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_react_epic.txt | tee out_log_pp_react_epic.txt
echo "Evaluation for EPIC react finished"

### Mahalanobis ash

# UCF
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_ash_ucf.txt | tee out_log_pp_maha_ash_ucf.txt
echo "Evaluation for UCF Mahalanobis+ash finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_ash_hac.txt | tee out_log_pp_maha_ash_hac.txt
echo "Evaluation for HAC Mahalanobis+ash finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_ash_epic.txt | tee out_log_pp_maha_ash_epic.txt
echo "Evaluation for EPIC Mahalanobis+ash finished"

### Mahalanobis react

# UCF
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_react_ucf.txt | tee out_log_pp_maha_react_ucf.txt
echo "Evaluation for UCF Mahalanobis react finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_react_hac.txt | tee out_log_pp_maha_react_hac.txt
echo "Evaluation for HAC Mahalanobis react finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maha_react_epic.txt | tee out_log_pp_maha_react_epic.txt
echo "Evaluation for EPIC Mahalanobis react finished"

### MaxLogit ash
# UCF
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_ash_ucf.txt | tee out_log_pp_maxlog_ash_ucf.txt
echo "Evaluation for UCF MaxLogit ash finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_ash_hac.txt | tee out_log_pp_maxlog_ash_hac.txt
echo "Evaluation for HAC MaxLogit ash finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_ash_epic.txt | tee out_log_pp_maxlog_ash_epic.txt
echo "Evaluation for EPIC MaxLogit ash finished"

### MaxLogit react
# UCF
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_react_ucf.txt | tee out_log_pp_maxlog_react_ucf.txt
echo "Evaluation for UCF MaxLogit react finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_react_hac.txt | tee out_log_pp_maxlog_react_hac.txt
echo "Evaluation for HAC MaxLogit react finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor maxlogit --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_maxlog_react_epic.txt | tee out_log_pp_maxlog_react_epic.txt
echo "Evaluation for EPIC MaxLogit react finished"

### MSP ash
# UCF
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_ash_ucf.txt | tee out_log_pp_msp_ash_ucf.txt
echo "Evaluation for UCF MSP ash finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_ash_hac.txt | tee out_log_pp_msp_ash_hac.txt
echo "Evaluation for HAC MSP ash finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_ash_epic.txt | tee out_log_pp_msp_ash_epic.txt
echo "Evaluation for EPIC MSP ash finished"

### MSP react
# UCF
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_react_ucf.txt | tee out_log_pp_msp_react_ucf.txt
echo "Evaluation for UCF MSP react finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_react_hac.txt | tee out_log_pp_msp_react_hac.txt
echo "Evaluation for HAC MSP react finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_msp_react_epic.txt | tee out_log_pp_msp_react_epic.txt
echo "Evaluation for EPIC MSP react finished"

### GEN ash

# UCF
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_ash_ucf.txt | tee out_log_pp_gen_ash_ucf.txt
echo "Evaluation for UCF GEN ash finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_ash_hac.txt | tee out_log_pp_gen_ash_hac.txt
echo "Evaluation for HAC GEN ash finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_ash_epic.txt | tee out_log_pp_gen_ash_epic.txt
echo "Evaluation for EPIC GEN ash finished"

### GEN react

# UCF
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_react_ucf.txt | tee out_log_pp_gen_react_ucf.txt
echo "Evaluation for UCF GEN react finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_react_hac.txt | tee out_log_pp_gen_react_hac.txt
echo "Evaluation for HAC GEN react finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_gen_react_epic.txt | tee out_log_pp_gen_react_epic.txt
echo "Evaluation for EPIC GEN react finished"

### KNN ash

# UCF
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_ash_ucf.txt | tee out_log_pp_knn_ash_ucf.txt
echo "Evaluation for UCF KNN ash finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_ash_hac.txt | tee out_log_pp_knn_ash_hac.txt
echo "Evaluation for HAC KNN ash finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_ash_epic.txt | tee out_log_pp_knn_ash_epic.txt
echo "Evaluation for EPIC KNN ash finished"

### KNN react

# UCF
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_react_ucf.txt | tee out_log_pp_knn_react_ucf.txt
echo "Evaluation for UCF KNN react finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_react_hac.txt | tee out_log_pp_knn_react_hac.txt
echo "Evaluation for HAC KNN react finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor knn --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_knn_react_epic.txt | tee out_log_pp_knn_react_epic.txt
echo "Evaluation for EPIC KNN react finished"

### EBO ash

# UCF
python eval_video_flow_far_ood.py --postprocessor ebo --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_ebo_ash_ucf.txt | tee out_log_pp_ebo_ash_ucf.txt
echo "Evaluation for UCF EBO ash finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor ebo --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_ebo_ash_hac.txt | tee out_log_pp_ebo_ash_hac.txt
echo "Evaluation for HAC EBO ash finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor ebo --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_ebo_ash_epic.txt | tee out_log_pp_ebo_ash_epic.txt
echo "Evaluation for EPIC EBO ash finished"

### EBO react

# UCF
python eval_video_flow_far_ood.py --postprocessor ebo --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_log_pp_ebo_react_ucf.txt | tee out_log_pp_ebo_react_ucf.txt
echo "Evaluation for UCF EBO react finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor ebo --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_log_pp_ebo_react_hac.txt | tee out_log_pp_ebo_react_hac.txt
echo "Evaluation for HAC EBO react finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor ebo --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_log_pp_ebo_react_epic.txt | tee out_log_pp_ebo_react_epic.txt
echo "Evaluation for EPIC EBO react finished"
