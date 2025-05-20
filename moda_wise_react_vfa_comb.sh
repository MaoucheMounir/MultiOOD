# Créer les fichiers d'évaluation en VFA avec des combinaisons de modalités

cd /data/maouche/MultiOOD/EPIC-rgb-flow
# ## Test
# # Créer les fichiers d'évaluation sans React
#python test_video_flow_audio_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa.log | tee out_test_EPIC_vfa.log
#"echo "Saved Evaluation Files for EPIC vfa \n"

# # Créer les fichiers d'évaluation avec React sur tout
# python test_video_flow_audio_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_react.log | tee out_test_EPIC_vfa_react.log
# echo "Saved Evaluation Files for EPIC vfa React \n"

# ########################################################

# # Créer les fichiers d'évaluation avec React sur Video et Flow 
# python test_video_flow_audio_moda_wise_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --drop_modality video_flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_react_video_flow.log | tee out_test_EPIC_vfa_react_video_flow.log
# echo "Saved Evaluation Files for EPIC vfa React video flow \n"

# # Créer les fichiers d'évaluation avec React sur Flow et Audio
# python test_video_flow_audio_moda_wise_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --drop_modality flow_audio --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_react_flow_audio.log | tee out_test_EPIC_vfa_react_flow_audio.log
# echo "Saved Evaluation Files for EPIC vfa React flow audio \n"

# # # Créer les fichiers d'évaluation avec React sur Video et Audio
# python test_video_flow_audio_moda_wise_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --use_react --drop_modality video_audio  --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_vfa_baseline.pt' 2>error_test_EPIC_vfa_react_video_audio.log | tee out_test_EPIC_vfa_react_video_audio.log
# echo "Saved Evaluation Files for EPIC vfa React video audio\n"

#########################################################
cd ..
# Eval
# Evaluation sans React
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_' --dataset 'EPIC' --path 'EPIC-rgb-flow/' 2>error_eval_EPIC_vfa.log | tee out_eval_EPIC_vfa.log
# echo "Evaluation for Epic vfa finished \n"

# # Evaluation avec React sur tout
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_react_' --dataset 'EPIC' --path 'EPIC-rgb-flow/' 2>error_eval_EPIC_vfa_react.log | tee out_eval_EPIC_vfa_react.log
# echo "Evaluation for Epic vfa react finished \n"

#########################################################
# Evaluation React video flow
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_react_video_flow_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_vfa_react_video_flow.log | tee out_eval_EPIC_vfa_react_video_flow.log
# echo "Evaluation for Epic vfa react video flow finished \n"

# Evaluation react flow audio
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_react_flow_audio_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_vfa_react_flow_audio.log | tee out_eval_EPIC_vfa_react_flow_audio.log
# echo "Evaluation for Epic vfa react flow audio finished \n"

# # Evaluation React video audio
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_baseline_best_react_video_audio_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_vfa_react_video_audio.log | tee out_eval_EPIC_vfa_react_video_audio.log
# echo "Evaluation for Epic vfa react video audio finished \n"
##########################################################

# python eval_video_flow_near_ood_moda_wise.py --aggregation mean --appen vfa_baseline_best_react_
# python eval_video_flow_near_ood_moda_wise.py --aggregation max --appen vfa_baseline_best_react_

##Ajouter le truc de comparer la perturbation pour choisir quelle modalité prendre