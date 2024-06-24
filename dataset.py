import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm
import nibabel as nib
import numpy as np
import random
import glob
import os
import re
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BraTS(Dataset):
    def __init__(self, path, mode='train', mask=False):
        # path: path to the dataset
        # mode: dataset mode ('train', 'test', or 'val')
        # mask: indicates whether to include segmentation masks

        # Splitting the dataset based on the mode
        if mode == 'train':
            tiles, _, _ = self.split(glob.glob(os.path.join(path, '*')))
        elif mode == 'test':
            _, tiles, _ = self.split(glob.glob(os.path.join(path, '*')))
        elif mode == 'val':
            _, _, tiles = self.split(glob.glob(os.path.join(path, '*')))
        else:
            raise ValueError("Invalid mode.")

        # Initializing lists for input images and targets
        self.X = []
        self.y = []

        # Iterating over dataset tiles
        for tile in tqdm(tiles, desc='Loading '+mode+' data'):
            # Extracting brain slice indices
            splits = [f.split('truth_')[-1] for f in glob.glob(os.path.join(tile, 'truth_*'))]

            # Iterating over each brain slice
            for i in [int(re.search(r'\d+', item).group()) for item in splits]:
                # Getting bands for each brain slice
                all_bands = glob.glob(os.path.join(tile, '*_'+str(i)+'.nii.gz'))
                # Excluding ground truth from the list
                all_bands = [file for file in all_bands if "truth_"+str(i)+".nii.gz" not in file]
                current_image = []

                # Checking if all 4 bands are present
                if len(all_bands) == 4:
                    # Loading images and appending to the list
                    for file_path in all_bands:
                        img = nib.load(file_path)
                        data = img.get_fdata()
                        current_image.append(data)

                    # Adding mask if required
                    if mask:
                        mas = np.where(current_image[0] != 0, 1, 0)
                        current_image.append(mas)
                    
                    # Stacking images
                    stacked_image = np.stack(current_image)
                    self.X.append(stacked_image)

                    # Loading ground truth and converting labels
                    ds = nib.load(os.path.join(tile, 'truth_'+str(i)+'.nii.gz')).get_fdata()
                    temp = np.copy(ds.astype(np.int64))
                    temp[temp == 4] = 3
                    self.y.append(temp)

        self.X = np.stack(self.X)
        self.y = np.stack(self.y)

        self.mode = mode
        self.length = len(self.y)
        self.path = path
        self.input_size = 240

    def __len__(self):
        # Return the length of the dataset
        return self.length

    def __getitem__(self, index):
        # Return an instance of the dataset
        image = self.X[index]
        target = self.y[index]

        image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype('float32')       # CxWxH to WxHxC
        target = target[:, :, np.newaxis]

        #Imagine Transformation
        if self.mode=='train':
            image, target = self.transform(image, target)
            
        image = image.astype(np.float32).transpose(2, 0, 1)
        target = target.squeeze()

        return image.copy(), target.copy()

    def split(self, tiles):
        # Randomly split the folders
        random.shuffle(tiles)

        # Split proportions
        train_ratio = 0.7
        test_ratio = 0.3
        validation_ratio = 0.3

        # Calculating the number of folders for each set
        total_folders = len(tiles)
        train_count = int(total_folders * train_ratio)
        test_count = int(total_folders * test_ratio)
        validation_count = int(train_count * validation_ratio)

        # Splitting the folders
        train_folders = tiles[:train_count]
        test_folders = tiles[train_count:train_count + test_count]
        validation_folders = train_folders[:validation_count]
        train_folders = train_folders[validation_count:]

        return train_folders, test_folders, validation_folders
    
    def transform(self, image, target):
        aug = A.Compose([
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=240, sigma=240 * 0.05, alpha_affine=240 * 0.03),
                A.GridDistortion(p=0.5)
            ], p=0.4)
        ])

        augmented = aug(image=image, mask=target)
        image_augmented = augmented['image']
        target_augmented = augmented['mask']
        return image_augmented, target_augmented