import os
import random
import shutil

def split_dataset(data_path, validation_num = 200):
    samples = os.listdir(data_path)
    random.shuffle(samples)

    train_filepaths = samples[:-validation_num]
    val_filepaths = samples[-validation_num:]

    return train_filepaths, val_filepaths

if __name__ == '__main__':

    data_path = '/mnt/cdisk/boux/data/seco'
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    # Create dirs
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    # Split dataset
    train_filepaths, val_filepaths = split_dataset(data_path)

    # Move files to dirs
    for f in train_filepaths:
        src = os.path.join(data_path, f)
        shutil.move(src, train_dir)
    
    for f in val_filepaths:
        src = os.path.join(data_path, f)
        shutil.move(src, val_dir)