python main.py --epochs 5 --workers 16 --batches 64 --lang "en" \
    --train /home/shawon/Projects/parser-data/english/train/wsj_train.conll06 \
    --dev /home/shawon/Projects/parser-data/english/dev/wsj_dev.conll06.gold \
    --pre ./persisted_data/preprocessed_en.json \
    --model "microsoft/xtremedistil-l6-h384-uncased"
