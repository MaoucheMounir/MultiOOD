# Fichier destiné à calculer les MSP avec différentes manières d'appliquer React
# Pour entraîner le classifieur dessus mais cette fois-ci avec 3 modalités.

# cd /data/maouche/MultiOOD/EPIC-rgb-flow
# ## Test
# # Créer les fichiers d'évaluation sans React
# python test_video_flow_audio_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa.log | tee out_test_EPIC_vfa.log
# echo "Saved Evaluation Files for EPIC vfa \n"

# # Créer les fichiers d'évaluation avec React sur tout
# python test_video_flow_audio_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_react.log | tee out_test_EPIC_vfa_react.log
# echo "Saved Evaluation Files for EPIC vfa React \n"

# ########################################################

# # Créer les fichiers d'évaluation avec React sur Video
# python test_video_flow_audio_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_react_video.log | tee out_test_EPIC_vfa_react_video.log
# echo "Saved Evaluation Files for EPIC vfa React video\n"

# # Créer les fichiers d'évaluation avec React sur Flow
# python test_video_flow_audio_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_react_flow.log | tee out_test_EPIC_vfa_react_flow.log
# echo "Saved Evaluation Files for EPIC vfa React flow \n"

# # Créer les fichiers d'évaluation avec React sur Audio
# python test_video_flow_audio_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --drop_modality audio  --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_react_audio.log | tee out_test_EPIC_vfa_react_audio.log
# echo "Saved Evaluation Files for EPIC vfa React audio\n"

#########################################################
#cd ..
# Eval
# Evaluation sans React
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_' --dataset 'EPIC' --path 'EPIC-rgb-flow/' 2>error_eval_EPIC_vfa.log | tee out_eval_EPIC_vfa.log
echo "Evaluation for Epic vfa finished \n"

# Evaluation avec React sur tout
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_react_' --dataset 'EPIC' --path 'EPIC-rgb-flow/' 2>error_eval_EPIC_vfa_react.log | tee out_eval_EPIC_vfa_react.log
echo "Evaluation for Epic vfa react finished \n"

#########################################################
# Evaluation sans React
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_react_video_' --dataset 'EPIC' --path 'EPIC-rgb-flow/' 2>error_eval_EPIC_vfa_react_video.log | tee out_eval_EPIC_vfa_react_video.log
echo "Evaluation for Epic vfa react video finished \n"

# Evaluation sans React
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_react_flow_' --dataset 'EPIC' --path 'EPIC-rgb-flow/' 2>error_eval_EPIC_vfa_react_flow.log | tee out_eval_EPIC_vfa_react_flow.log
echo "Evaluation for Epic vfa react flow finished \n"

# Evaluation sans React
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_react_audio_' --dataset 'EPIC' --path 'EPIC-rgb-flow/' 2>error_eval_EPIC_vfa_react_audio.log | tee out_eval_EPIC_vfa_react_audio.log
echo "Evaluation for Epic vfa react audio finished \n"
##########################################################