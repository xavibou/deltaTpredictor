import os
import torch
import random
from pathlib import Path
import torchvision
import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import datetime
from pytorch_histogram_matching import Histogram_Matching
#from pl_bolts.models.self_supervised.moco.transforms import GaussianBlur, imagenet_normalization

ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']
RGBNir_BANDS = ['B4', 'B3', 'B2', 'B8']

QUANTILES = {
    'min_q': {
        'B2': 3.0,
        'B3': 2.0,
        'B4': 0.0
    },
    'max_q': {
        'B2': 88.0,
        'B3': 103.0,
        'B4': 129.0
    }
}

class DeltaTimeBase(Dataset):

    def __init__(self, root, bands=None, transform=None):
        super().__init__()
        self.root = Path(root)
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform
        self.HM = Histogram_Matching(differentiable=True)

        self._samples = None

    @property
    def samples(self):
        if self._samples is None:
            self._samples = self.get_samples()
        return self._samples

    def get_samples(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def read_image(path, bands, quantiles=None):
    channels = []
    for b in bands:
        ch = rasterio.open(path / f'{b}.tif').read(1)
        if quantiles is not None:
            ch = normalize(ch, min_q=quantiles['min_q'][b], max_q=quantiles['max_q'][b])
        channels.append(ch)
    img = np.dstack(channels)
    img = Image.fromarray(img)
    return img

class DeltaTimeDataset(DeltaTimeBase):

    '''augment = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        #transforms.RandomHorizontalFlip(),
    ])'''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    def get_samples(self):
        return [path for path in self.root.glob('*') if path.is_dir()]

    def get_date(self, path):
        '''
        Given a path to a satellite image, return the date of the image
        Args:
            path: path to satellite image
        '''
        path = str(path).split("/")[-1]
        year = int(path[:4])
        month = int(path[4:6])
        day = int(path[6:8])
        return datetime.date(year, month, day)

    def get_delta_t(self, pathList):
        ''' 
        Given two satellite image paths, return the delta_t between them in days
        Args:
            pathList: list of two paths to satellite images
        '''
        
        # Get dates from paths
        date1 = self.get_date(pathList[0])
        date2 = self.get_date(pathList[1])

        # Compute and return delta_t in days
        delta_t = abs(date2 - date1)

        '''print()
        print("Date 1: ", date1)
        print("Date 2: ", date2)
        print("Delta t: ", delta_t)
        print(pathList[0])
        print(pathList[1])
        print()'''

        return np.array(delta_t.days / 365.0)

    def __getitem__(self, index):
        root = self.samples[index]

        # Extract two random images from the series
        sorted_paths = sorted([path for path in root.glob('*') if path.is_dir()], reverse=True)
        if len(sorted_paths) < 2:
            sorted_paths = [sorted_paths[0], sorted_paths[0]]
        #rand_paths = np.random.choice(sorted_paths, 2, replace=False)
        np.random.shuffle(sorted_paths)
        rand_paths = sorted_paths[:2]
        t1, t2 = [read_image(path, self.bands, QUANTILES) for path in rand_paths]

        delta_t = self.get_delta_t(rand_paths)
        delta_t = torch.from_numpy(delta_t)

        # Random Augmentations
        #t1 = self.augment(t1)
        #t2 = self.augment(t2)

        # Preprocess
         # Apply the same random crop to query image and image to be reconstructed by the decoder
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img transforms
        torch.manual_seed(seed)
        t1 = self.preprocess(t1)

        torch.manual_seed(seed)
        t2 = self.preprocess(t2)

        #t2 = self.HM(t2[None,...], t1[None,...]).squeeze()

        '''print()
        print()
        print(t1.shape)
        print(t2.shape)
        print()
        print()'''

        return t1, t2, delta_t

if __name__ == '__main__':

    def save_tensor_as_image(tensor, file_path):
        # Normalize tensor values to [0, 1]
        tensor = tensor.clamp(0, 1)
        # Convert tensor to PIL image
        image = torchvision.transforms.ToPILImage()(tensor)
        # Save PIL image to file path
        image.save(file_path)
    
    
    data_path = '/mnt/cdisk/boux/data/seco'

    dataset = DeltaTimeDataset(data_path)
    t1, t2, delta_t = dataset[200]
    print()
    print(delta_t)

    print(t1.shape)
    print(t2.shape)

    saver_dir = '/mnt/cdisk/boux/code/time_diff_prediction/image_tests'
    save_tensor_as_image(t1, os.path.join(saver_dir, 't1.png'))
    save_tensor_as_image(t2, os.path.join(saver_dir, 't2.png'))