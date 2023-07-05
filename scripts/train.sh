
python train.py \
    --data_dir /mnt/ddisk/boux/code/data/seco/test_small_dataset \
    --base_encoder resnet50 \
    --num_workers 1 \
    --batch_size 1 \
    --learning_rate 0.01 \
    --max_epochs 200 \
    --task regression \
    