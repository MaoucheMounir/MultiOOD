# # # Créer les fichiers d'évaluation sans ash
# #python test_video_flow_audio_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa.log | tee out_test_EPIC_vfa.log
# #echo "Saved Evaluation Files for EPIC vfa \n"
# cd "EPIC-rgb-flow"
# # # # Créer les fichiers d'évaluation avec ash sur tout
# # python test_video_flow_audio_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_ash --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_ash.log | tee out_test_EPIC_vfa_ash.log
# # echo "Saved Evaluation Files for EPIC vfa ash \n"

# # python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_ash_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_vfa_ash.log | tee out_eval_EPIC_vfa_ash.log
# # echo "Evaluation for Epic vfa ash finished \n"


# # Créer les fichiers d'évaluation avec ash sur Video
# python test_video_flow_audio_moda_wise_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_ash --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_ash_video.log | tee out_test_EPIC_vfa_ash_video.log
# echo "Saved Evaluation Files for EPIC vfa ash video\n"

# # Créer les fichiers d'évaluation avec ash sur Flow
# python test_video_flow_audio_moda_wise_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_ash --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_ash_flow.log | tee out_test_EPIC_vfa_ash_flow.log
# echo "Saved Evaluation Files for EPIC vfa ash flow \n"

# # Créer les fichiers d'évaluation avec ash sur Audio
# python test_video_flow_audio_moda_wise_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_ash --drop_modality audio  --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_ash_audio.log | tee out_test_EPIC_vfa_ash_audio.log
# echo "Saved Evaluation Files for EPIC vfa ash audio\n"


# cd ..

# #########################################################
# # Evaluation ash video
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_ash_video_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_vfa_ash_video.log | tee out_eval_EPIC_vfa_ash_video.log
# echo "Evaluation for Epic vfa ash video finished \n"

# # Evaluation ash flow
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_ash_flow_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_vfa_ash_flow.log | tee out_eval_EPIC_vfa_ash_flow.log
# echo "Evaluation for Epic vfa ash flow finished \n"

# # Evaluation ash audio
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_ash_audio_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_vfa_ash_audio.log | tee out_eval_EPIC_vfa_ash_audio.log
# echo "Evaluation for Epic vfa ash audio finished \n"
##########################################################

python eval_video_flow_near_ood_moda_wise.py --appen baseline_best_ash_ --vfa03333.25...7