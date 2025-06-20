# Généré par : script_near.py
# cd /data/maouche/MultiOOD/HMDB-rgb-flow
# python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_ash --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_baseline.pt' 2>error_test_HMDB_near_ood_ash_video.log | tee out_test_HMDB_near_ood_ash_video.log
# echo "Saved evaluation files for HMDB ash_ video"
# cd ..
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_ash_video_' --dataset 'HMDB' --path 'HMDB-rgb-flow/' 2>error_eval_HMDB_near_ood_ash_video.log | tee out_eval_HMDB_near_ood_ash_video.log
# echo "Evaluation for HMDB ash_ video finished"

# cd /data/maouche/MultiOOD/HMDB-rgb-flow
# python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_ash --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_baseline.pt' 2>error_test_UCF_near_ood_ash_video.log | tee out_test_UCF_near_ood_ash_video.log
# echo "Saved evaluation files for UCF ash_ video"
# cd ..
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_ash_video_' --dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_eval_UCF_near_ood_ash_video.log | tee out_eval_UCF_near_ood_ash_video.log
# echo "Evaluation for UCF ash_ video finished"

# cd /data/maouche/MultiOOD/EPIC-rgb-flow
# python test_video_flow_moda_wise_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --dataset 'EPIC' --use_ash --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_baseline.pt' 2>error_test_EPIC_near_ood_ash_video.log | tee out_test_EPIC_near_ood_ash_video.log
# echo "Saved evaluation files for EPIC ash_ video"
# cd ..
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_ash_video_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_near_ood_ash_video.log | tee out_eval_EPIC_near_ood_ash_video.log
# echo "Evaluation for EPIC ash_ video finished"

# cd /data/maouche/MultiOOD/HMDB-rgb-flow
# python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_ash --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_baseline.pt' 2>error_test_HMDB_near_ood_ash_flow.log | tee out_test_HMDB_near_ood_ash_flow.log
# echo "Saved evaluation files for HMDB ash_ flow"
# cd ..
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_ash_flow_' --dataset 'HMDB' --path 'HMDB-rgb-flow/' 2>error_eval_HMDB_near_ood_ash_flow.log | tee out_eval_HMDB_near_ood_ash_flow.log
# echo "Evaluation for HMDB ash_ flow finished"

# cd /data/maouche/MultiOOD/HMDB-rgb-flow
# python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_ash --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_baseline.pt' 2>error_test_UCF_near_ood_ash_flow.log | tee out_test_UCF_near_ood_ash_flow.log
# echo "Saved evaluation files for UCF ash_ flow"
# cd ..
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_ash_flow_' --dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_eval_UCF_near_ood_ash_flow.log | tee out_eval_UCF_near_ood_ash_flow.log
# echo "Evaluation for UCF ash_ flow finished"

# cd /data/maouche/MultiOOD/EPIC-rgb-flow
# python test_video_flow_moda_wise_epic.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --dataset 'EPIC' --use_ash --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_baseline.pt' 2>error_test_EPIC_near_ood_ash_flow.log | tee out_test_EPIC_near_ood_ash_flow.log
# echo "Saved evaluation files for EPIC ash_ flow"
# cd ..
# python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_ash_flow_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_near_ood_ash_flow.log | tee out_eval_EPIC_near_ood_ash_flow.log
# echo "Evaluation for EPIC ash_ flow finished"

python eval_video_flow_near_ood_moda_wise.py --aggregation mean --appen baseline_best_ash_
python eval_video_flow_near_ood_moda_wise.py --aggregation max --appen baseline_best_ash_

