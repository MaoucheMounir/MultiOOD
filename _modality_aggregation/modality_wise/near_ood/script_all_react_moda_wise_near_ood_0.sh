cd /data/maouche/MultiOOD/HMDB-rgb-flow
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_react --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_baseline.pt' 2>error_test_HMDB_react_video.log | tee out_test_HMDB_react_video.log
echo "Saved evaluation files for HMDB react_ video"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'HMDB'--path 'HMDB-rgb-flow/' 2>error_eval_HMDB_react_video.log | tee out_eval_HMDB_react_video.log
echo "Evaluation for HMDB react_ video finished"

cd /data/maouche/MultiOOD/HMDB-rgb-flow
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_react --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_baseline.pt' 2>error_test_UCF_react_video.log | tee out_test_UCF_react_video.log
echo "Saved evaluation files for UCF react_ video"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'UCF'--path 'HMDB-rgb-flow/' 2>error_eval_UCF_react_video.log | tee out_eval_UCF_react_video.log
echo "Evaluation for UCF react_ video finished"

cd /data/maouche/MultiOOD/EPIC-rgb-flow
python test_video_flow_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --near_ood --dataset 'EPIC' --use_react --drop_modality video --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rg-flow/EPIC_near_ood_baseline.pt' 2>error_test_EPIC_react_video.log | tee out_test_EPIC_react_video.log
echo "Saved evaluation files for EPIC react_ video"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'EPIC'--path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_video.log | tee out_eval_EPIC_react_video.log
echo "Evaluation for EPIC react_ video finished"

cd /data/maouche/MultiOOD/HMDB-rgb-flow
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_react --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_baseline.pt' 2>error_test_HMDB_react_flow.log | tee out_test_HMDB_react_flow.log
echo "Saved evaluation files for HMDB react_ flow"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'HMDB'--path 'HMDB-rgb-flow/' 2>error_eval_HMDB_react_flow.log | tee out_eval_HMDB_react_flow.log
echo "Evaluation for HMDB react_ flow finished"

cd /data/maouche/MultiOOD/HMDB-rgb-flow
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_react --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_baseline.pt' 2>error_test_UCF_react_flow.log | tee out_test_UCF_react_flow.log
echo "Saved evaluation files for UCF react_ flow"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'UCF'--path 'HMDB-rgb-flow/' 2>error_eval_UCF_react_flow.log | tee out_eval_UCF_react_flow.log
echo "Evaluation for UCF react_ flow finished"

cd /data/maouche/MultiOOD/EPIC-rgb-flow
python test_video_flow_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --near_ood --dataset 'EPIC' --use_react --drop_modality flow --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rg-flow/EPIC_near_ood_baseline.pt' 2>error_test_EPIC_react_flow.log | tee out_test_EPIC_react_flow.log
echo "Saved evaluation files for EPIC react_ flow"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'EPIC'--path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_flow.log | tee out_eval_EPIC_react_flow.log
echo "Evaluation for EPIC react_ flow finished"

cd /data/maouche/MultiOOD/HMDB-rgb-flow
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --use_react --drop_modality audio --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_near_ood_baseline.pt' 2>error_test_HMDB_react_audio.log | tee out_test_HMDB_react_audio.log
echo "Saved evaluation files for HMDB react_ audio"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_audio_' --dataset 'HMDB'--path 'HMDB-rgb-flow/' 2>error_eval_HMDB_react_audio.log | tee out_eval_HMDB_react_audio.log
echo "Evaluation for HMDB react_ audio finished"

cd /data/maouche/MultiOOD/HMDB-rgb-flow
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/UCF101/ --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --use_react --drop_modality audio --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/UCF_near_ood_baseline.pt' 2>error_test_UCF_react_audio.log | tee out_test_UCF_react_audio.log
echo "Saved evaluation files for UCF react_ audio"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_audio_' --dataset 'UCF'--path 'HMDB-rgb-flow/' 2>error_eval_UCF_react_audio.log | tee out_eval_UCF_react_audio.log
echo "Evaluation for UCF react_ audio finished"

cd /data/maouche/MultiOOD/EPIC-rgb-flow
python test_video_flow_epic_moda_wise.py --datapath /data/maouche/MultiOOD/EPIC-KITCHENS/ --bsz 16 --num_workers 2 --near_ood --dataset 'EPIC' --use_react --drop_modality audio --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/EPIC-rg-flow/EPIC_near_ood_baseline.pt' 2>error_test_EPIC_react_audio.log | tee out_test_EPIC_react_audio.log
echo "Saved evaluation files for EPIC react_ audio"
cd ..
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_audio_' --dataset 'EPIC'--path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_audio.log | tee out_eval_EPIC_react_audio.log
echo "Evaluation for EPIC react_ audio finished"

