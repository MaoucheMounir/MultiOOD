shopt -s globstar
for file in /data/maouche/MultiOOD/UCF101/video/UCF-101/**/**/*.mp4; do
    mv "$file" /data/maouche/MultiOOD/UCF101/video
done