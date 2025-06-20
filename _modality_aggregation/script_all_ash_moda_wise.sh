cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use-ash --drop_modality video --ood_dataset 'HAC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_HAC_ash_video.log | tee out_test_HAC_ash_video.log
echo "Saved evaluation files for HAC ash_ video"
cd ..
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_ash_video_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_eval_HAC_ash_video.log | tee out_eval_HAC_ash_video.log
echo "Evaluation for HAC ash_ video finished"

cd "/data/maouche/MultiOOD/HMDB-rgb-flow"
python test_video_flow_moda_wise.py --datapath /data/maouche/MultiOOD/HAC/ --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --use-ash --drop_modality flow --ood_dataset 'HAC' --appen 'baseline_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_baseline.pt' 2>error_test_HAC_ash_flow.log | tee out_test_HAC_ash_flow.log
echo "Saved evaluation files for HAC ash_ flow"
cd ..
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseline_best_ash_flow_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/' 2>error_eval_HAC_ash_flow.log | tee out_eval_HAC_ash_flow.log
echo "Evaluation for HAC ash_ flow finished"

