cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --use_react --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_HMDB_react_video.log | tee out_test_HMDB_react_video.log
echo "Saved evaluation files for HMDB react_ video"

python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --use_react --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_HMDB_react_flow.log | tee out_test_HMDB_react_flow.log
echo "Saved evaluation files for HMDB react_ flow"

cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --drop_modality video --ood_dataset 'UCF' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_UCF_react_video.log | tee out_test_UCF_react_video.log
echo "Saved evaluation files for UCF react_ video"
cd ..
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_eval_UCF_react_video.log | tee out_eval_UCF_react_video.log
echo "Evaluation for UCF react_ video finished"

cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --drop_modality video --ood_dataset 'HAC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_HAC_react_video.log | tee out_test_HAC_react_video.log
echo "Saved evaluation files for HAC react_ video"
cd ..
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_eval_HAC_react_video.log | tee out_eval_HAC_react_video.log
echo "Evaluation for HAC react_ video finished"

cd "/data/maouche/MultiOOD/EPIC-rgb-flow"
python test_video_flow_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --drop_modality video --ood_dataset 'EPIC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_EPIC_react_video.log | tee out_test_EPIC_react_video.log
echo "Saved evaluation files for EPIC react_ video"
cd ..
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_video.log | tee out_eval_EPIC_react_video.log
echo "Evaluation for EPIC react_ video finished"

cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --drop_modality flow --ood_dataset 'UCF' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_UCF_react_flow.log | tee out_test_UCF_react_flow.log
echo "Saved evaluation files for UCF react_ flow"
cd ..
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_eval_UCF_react_flow.log | tee out_eval_UCF_react_flow.log
echo "Evaluation for UCF react_ flow finished"

cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --drop_modality flow --ood_dataset 'HAC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_HAC_react_flow.log | tee out_test_HAC_react_flow.log
echo "Saved evaluation files for HAC react_ flow"
cd ..
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_eval_HAC_react_flow.log | tee out_eval_HAC_react_flow.log
echo "Evaluation for HAC react_ flow finished"

cd "/data/maouche/MultiOOD/EPIC-rgb-flow"
python test_video_flow_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use_react --drop_modality flow --ood_dataset 'EPIC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_EPIC_react_flow.log | tee out_test_EPIC_react_flow.log
echo "Saved evaluation files for EPIC react_ flow"
cd ..
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_flow.log | tee out_eval_EPIC_react_flow.log
echo "Evaluation for EPIC react_ flow finished"

