for f in Kinetics-Video/*.tar.gz; do tar -xvf "$f" -C Kinetics-Video; done
python utils/generate_audio_files.py