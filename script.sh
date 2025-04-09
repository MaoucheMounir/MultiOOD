cd "HMDB-rgb-flow/"

# Save the evaluation files for HMDB 
# (to save evaluation files for ASH or ReAct, you should also run following line with options --use_ash or --use_react, same for other datasets):
python test_video_flow.py --datapath /data/maouche/MultiOOD/HMDB51/ --bsz 16 --num_workers 2 --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_a2d_npmix.pt' 2>error_log.txt | tee out_log.txt
