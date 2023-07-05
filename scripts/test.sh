
python test.py \
    --data_dir /mnt/ddisk/boux/code/data/seco/test_small_dataset \
    --base_encoder resnet50 \
    --ckpt_path /mnt/ddisk/boux/code/deltaTpredictor/logs/0to7_deltaT_regression/resnet50-test_small_dataset-epochs=200/version_18/checkpoints/epoch=199.ckpt \
    --num_workers 1 \
    --batch_size 2 \
    --task regression \
    