import os
from argparse import ArgumentParser
import warnings
warnings.simplefilter('ignore', UserWarning)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.datamodule import DeltaTimeDatamodule
from models.model import DeltaTPredictor, DeltaTClassifier

def get_experiment_name(hparams):
    data_name = os.path.basename(hparams.data_dir)
    name = f'{hparams.base_encoder}-{data_name}-epochs={hparams.max_epochs}'
    return name


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = DeltaTPredictor.add_model_specific_args(parser)
    parser = ArgumentParser(parents=[parser], conflict_handler='resolve', add_help=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--online_data_dir', type=str)
    parser.add_argument('--online_max_epochs', type=int, default=25)
    parser.add_argument('--online_val_every_n_epoch', type=int, default=25)
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    datamodule = DeltaTimeDatamodule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.task == 'regression':
        model = DeltaTPredictor(**vars(args))
    elif args.task == 'classification':
        model = DeltaTClassifier(**vars(args))
    else:
        raise ValueError(f'Unknown task: {args.task}')

    if args.debug:
        logger = False
        checkpoint_callback = False
    else:
        logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), 'logs', '0to7_deltaT_{}'.format(args.task)),
            name=get_experiment_name(args)
        )
        checkpoint_callback = ModelCheckpoint(filename='{epoch}')

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
        weights_summary='full'
    )
    trainer.fit(model, datamodule=datamodule)
