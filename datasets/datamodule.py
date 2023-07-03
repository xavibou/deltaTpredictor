from pytorch_lightning import LightningDataModule
from datasets.dataset import DeltaTimeDataset
from utils.data import InfiniteDataLoader


class DeltaTimeBaseDataModule(LightningDataModule):

    def __init__(self, data_dir, bands=None, batch_size=32, num_workers=8, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None

    def setup(self, stage=None):
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        self.train_dataset = self.get_dataset(root=self.data_dir, bands=self.bands, transform=train_transforms)

    @staticmethod
    def get_dataset(root, bands, transform):
        raise NotImplementedError

    @staticmethod
    def train_transform():
        raise NotImplementedError

    def train_dataloader(self):
        return InfiniteDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

class DeltaTimeDatamodule(DeltaTimeBaseDataModule):

    @staticmethod
    def get_dataset(root, bands, transform):
        return DeltaTimeDataset(root, bands, transform)

    @staticmethod
    def train_transform():
        return None
