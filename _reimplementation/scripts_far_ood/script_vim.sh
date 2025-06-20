#--resume_file checkpoint.pt, where checkpoint.pt is the trained checkpoint
### Vim vanilla
# UCF
python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_ucf.txt | tee out_log_pp_vim_ucf.txt
echo "Evaluation for UCF vim finished"

# HAC
python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_hac.txt | tee out_log_pp_vim_hac.txt
echo "Evaluation for HAC vim finished"

# EPIC
python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_epic.txt | tee out_log_pp_vim_epic.txt
echo "Evaluation for EPIC vim finished"

# ### VIM ash

# # UCF
# python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_ash_ucf.txt | tee out_log_pp_vim_ash_ucf.txt
# echo "Evaluation for UCF VIM ash finished"

# # HAC
# python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_ash_hac.txt | tee out_log_pp_vim_ash_hac.txt
# echo "Evaluation for HAC VIM ash finished"

# # EPIC
# python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_ash_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_ash_epic.txt | tee out_log_pp_vim_ash_epic.txt
# echo "Evaluation for EPIC VIM ash finished"

# ### VIM react

# # UCF
# python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_react_ucf.txt | tee out_log_pp_vim_react_ucf.txt
# echo "Evaluation for UCF VIM react finished"

# # HAC
# python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_react_hac.txt | tee out_log_pp_vim_react_hac.txt
# echo "Evaluation for HAC VIM react finished"

# # EPIC
# python eval_video_flow_far_ood.py --postprocessor vim --appen 'a2d_npmix_best_react_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' --resume_file /data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt 2>error_log_pp_vim_react_epic.txt | tee out_log_pp_vim_react_epic.txt
# echo "Evaluation for EPIC VIM react finished"
