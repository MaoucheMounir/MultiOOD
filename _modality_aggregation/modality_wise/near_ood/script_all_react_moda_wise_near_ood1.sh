python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'HMDB' --path 'HMDB-rgb-flow/' 2>error_eval_HMDB_react_video.log | tee out_eval_HMDB_react_video.log
echo "Evaluation for HMDB react_ video finished"

python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_eval_UCF_react_video.log | tee out_eval_UCF_react_video.log
echo "Evaluation for UCF react_ video finished"

python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_video_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_video.log | tee out_eval_EPIC_react_video.log
echo "Evaluation for EPIC react_ video finished"

python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'HMDB' --path 'HMDB-rgb-flow/' 2>error_eval_HMDB_react_flow.log | tee out_eval_HMDB_react_flow.log
echo "Evaluation for HMDB react_ flow finished"

python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_eval_UCF_react_flow.log | tee out_eval_UCF_react_flow.log
echo "Evaluation for UCF react_ flow finished"

python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_flow_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_flow.log | tee out_eval_EPIC_react_flow.log
echo "Evaluation for EPIC react_ flow finished"

python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_audio_' --dataset 'HMDB' --path 'HMDB-rgb-flow/' 2>error_eval_HMDB_react_audio.log | tee out_eval_HMDB_react_audio.log
echo "Evaluation for HMDB react_ audio finished"

python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_audio_' --dataset 'UCF' --path 'HMDB-rgb-flow/' 2>error_eval_UCF_react_audio.log | tee out_eval_UCF_react_audio.log
echo "Evaluation for UCF react_ audio finished"

python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseline_best_react_audio_' --dataset 'EPIC' --path 'HMDB-rgb-flow/' 2>error_eval_EPIC_react_audio.log | tee out_eval_EPIC_react_audio.log
echo "Evaluation for EPIC react_ audio finished"

