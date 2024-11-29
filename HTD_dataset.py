import os
from cv2 import IMWRITE_PAM_FORMAT_GRAYSCALE
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as spo
import cv2
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class HTD_dataset(Dataset):
    def __init__(self, img_dir, img_name, target_abu, img_noise=None, prior_transform=['dialation']):
        img_path = os.path.join(img_dir, img_name + '.mat')
        data = spo.loadmat(img_path)
        self.img = data['img'].T.astype(np.float64)#(h*w, bands)
        self.groundtruth = data['groundtruth'].astype(np.uint8)

        if 'prior' in data.keys():
            prior = data['prior'][0,:]
        else:
            if 'dialation' in prior_transform:
                kernel = np.ones((2,2), np.uint8)
                pure_pixel = data['groundtruth']
                pure_pixel = cv2.erode(pure_pixel, kernel, iterations=1)
                prior = self.img[pure_pixel.reshape(-1, order='F')>0].mean(axis=0)
            else:
                pure_pixel = data['groundtruth']
                prior = self.img[pure_pixel.reshape(-1, order='F') > 0].mean(axis=0)
        self.prior = prior
        self.prior_norm = np.linalg.norm(prior, 2)
        self.target_abu = target_abu

        self.train_img = self.img

        self.test_img = self.img
        self.vd_num = 0


    def __len__(self):
        return self.train_img.shape[0]

    def __getitem__(self, idx):
        spectra = self.train_img[idx]
        target_abu = np.random.rand() * self.target_abu
        pseudo_target = spectra / np.linalg.norm(spectra, 2) * self.prior_norm * target_abu + self.prior * (1 - target_abu)
        return torch.Tensor(pseudo_target).cpu(), torch.Tensor(spectra).cpu()