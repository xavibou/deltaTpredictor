from argparse import ArgumentParser
from itertools import chain
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule

class DeltaTPredictor(LightningModule):

    def __init__(self, base_encoder, num_negatives, emb_spaces=1, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        template_model = getattr(torchvision.models, base_encoder)
        self.encoder = template_model()
        self.encoder.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,bias=False)  

        # remove fc layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1], nn.Flatten())

        # create the projection heads
        self.mlp_dim = 512 * (1 if base_encoder in ['resnet18', 'resnet34'] else 4)
        self.head = nn.Sequential(nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, 1))

        self.loss = nn.MSELoss()

    def forward(self, img_1, img_2):

        # Concatenate images into one tensor
        input = torch.cat((img_1, img_2), dim=1)

        x = self.encoder(input)     # Extract features
        x = self.head(x)            # Predict delta_t
        return x.type(torch.float64)

    def training_step(self, batch, batch_idx):
        img_1, img_2, delta_t = batch
        #delta_t = type(torch.float32)
        # Forward pass
        out = self(img_1, img_2)

        # Compute loss
        loss = self.loss(out, delta_t)

        # Logging to TensorBoard by default
        log = {'train_loss': loss}
        self.log_dict(log, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = chain(self.encoder.parameters(), self.head.parameters())
        optimizer = optim.SGD(params, self.hparams.learning_rate,
                              momentum=self.hparams.momentum,
                              weight_decay=self.hparams.weight_decay)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--base_encoder', type=str, default='resnet18')
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--num_negatives', type=int, default=16384)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=256)
        return parser

