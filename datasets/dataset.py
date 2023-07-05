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
from skimage.exposure import match_histograms

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
        #self.HM = Histogram_Matching(differentiable=True)

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

        return np.array(delta_t.days / (7*365.0))

    def __getitem__(self, index, return_unmatched_t2=False):
        root = self.samples[index]

        # Extract two random images from the series
        sorted_paths = sorted([path for path in root.glob('*') if path.is_dir()], reverse=True)
        rand_paths = random.sample(sorted_paths, 2)        
        try:
            t1, t2 = [read_image(path, self.bands, QUANTILES) for path in rand_paths]
        except:
            print(rand_paths)
            raise Exception("Error reading sample: ", root)

        # Match histograms
        t2_matched = match_histograms(np.array(t2), np.array(t1), channel_axis=0)
        t2_matched = Image.fromarray(t2_matched)

        delta_t = self.get_delta_t(rand_paths)
        delta_t = torch.from_numpy(delta_t)

        # Preprocess
        # Apply the same random crop to query image and image to be reconstructed by the decoder
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img transforms
        torch.manual_seed(seed)
        t1 = self.preprocess(t1)

        torch.manual_seed(seed)
        t2_matched = self.preprocess(t2_matched)

        if return_unmatched_t2:
            torch.manual_seed(seed)
            t2 = self.preprocess(t2)
            return t1, t2, t2_matched, delta_t

        return t1, t2_matched, delta_t

    
    def save_sample(self, save_dir, index=None, contrast_coeff=0.3, save_unmatched_t2=False):

        if index is None:
            index = np.random.randint(len(self))
        
        if save_unmatched_t2:
            t1, t2, unmatched_t2, delta_t = self.__getitem__(index, return_unmatched_t2=save_unmatched_t2)
        else:
            t1, t2, delta_t = self[index]

        t1 = t1.permute(1, 2, 0).numpy().squeeze()
        t2 = t2.permute(1, 2, 0).numpy().squeeze()

        t1 = (255 * np.clip(contrast_coeff + t1, 0, 1)).astype(np.uint8)
        t2 = (255 * np.clip(contrast_coeff + t2, 0, 1)).astype(np.uint8)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        Image.fromarray(t1).save(os.path.join(save_dir, 't1.png'))
        Image.fromarray(t2).save(os.path.join(save_dir, 't2.png'))

        if save_unmatched_t2:
            unmatched_t2 = unmatched_t2.permute(1, 2, 0).numpy().squeeze()
            unmatched_t2 = (255 * np.clip(contrast_coeff + unmatched_t2, 0, 1)).astype(np.uint8)
            Image.fromarray(unmatched_t2).save(os.path.join(save_dir, 't2_unmatched.png'))

        return t1, t2, delta_t

if __name__ == '__main__':
    
    data_path = '/mnt/ddisk/boux/code/data/seco/seco_intraseasonal/train'
    saver_dir = '/mnt/ddisk/boux/code/deltaTpredictor/image_tests'
    idx = None

    dataset = DeltaTimeDataset(data_path)
    t1, t2, delta_t = dataset.save_sample(saver_dir, index=idx, save_unmatched_t2=True)
    print()
    print('Delta T: {}'.format(delta_t.item()))