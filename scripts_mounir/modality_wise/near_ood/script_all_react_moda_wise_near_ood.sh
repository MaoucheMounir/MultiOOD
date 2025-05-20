cd /data/maouche/MultiOOD/EPIC-rgb-flow
python test_video_flow_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --dataset 'EPIC' --use_react --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_baseline.pt' 2>error_test_EPIC_react_video.log | tee out_test_EPIC_react_video.log
echo "Saved evaluation files for EPIC react_ video"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_video.log | tee out_eval_EPIC_react_video.log
echo "Evaluation for EPIC react_ video finished"


cd /data/maouche/MultiOOD/EPIC-rgb-flow
python test_video_flow_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --dataset 'EPIC' --use_react --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rgb-flow/EPIC_near_ood_baseline.pt' 2>error_test_EPIC_react_flow.log | tee out_test_EPIC_react_flow.log
echo "Saved evaluation files for EPIC react_ flow"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_flow.log | tee out_eval_EPIC_react_flow.log
echo "Evaluation for EPIC react_ flow finished"
