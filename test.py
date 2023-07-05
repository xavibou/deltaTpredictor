import os
import torch
from argparse import ArgumentParser
import warnings
warnings.simplefilter('ignore', UserWarning)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.datamodule import DeltaTimeDatamodule
from models.model import DeltaTPredictor, DeltaTClassifier
from utils.visualization_utils import plot_results

import numpy as np

def get_experiment_name(hparams):
    data_name = os.path.basename(hparams.data_dir)
    name = f'{hparams.base_encoder}-{data_name}-epochs={hparams.max_epochs}'
    return name

def generate_plot(model, dataloader):

    pred = []
    gt = []
    
    for img1, img2, delta_t in dataloader:
        img1 = img1.cuda()
        img2 = img2.cuda()

        with torch.no_grad():
            pred_delta_t = model(img1, img2)
        delta_t = delta_t.squeeze().cpu().detach().numpy().tolist()
        pred_delta_t = pred_delta_t.squeeze().cpu().detach().numpy().tolist()

        pred = pred + pred_delta_t
        gt = gt + delta_t

    plot_results(pred, gt)
    print("Predicted max/min: ", np.max(pred), np.min(pred))
    print("GT max/min: ", np.max(gt), np.min(gt))
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = DeltaTPredictor.add_model_specific_args(parser)
    parser = ArgumentParser(parents=[parser], conflict_handler='resolve', add_help=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    datamodule = DeltaTimeDatamodule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    datamodule.setup()


    if args.task == 'regression':
        model = DeltaTPredictor.load_from_checkpoint(args.ckpt_path)
    elif args.task == 'classification':
        model = DeltaTClassifier.load_from_checkpoint(args.ckpt_path)
    else:
        raise ValueError(f'Unknown task: {args.task}')

    model = model.cuda()
    generate_plot(model, datamodule.val_dataloader())