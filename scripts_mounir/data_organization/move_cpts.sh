for f in *.pth; do
    cp "$f" HMDB-rgb-flow/pretrained_models/
    cp "$f" EPIC-rgb-flow/pretrained_models/
done