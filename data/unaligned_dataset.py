import os
import sys
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from pathlib import Path

class UnalignedDataset(BaseDataset):
    """
    This dataset_aligned class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset_aligned flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, data_folder):
        """Initialize this dataset_aligned class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.tile_folders = sorted(
            [str(file) for file in Path(data_folder).glob('*')])
        self.image_folders_HE = []
        self.image_folders_MUC = []

        for tile_folder in self.tile_folders:
            image_folders_HE_current = sorted([str(file) for file in Path(tile_folder).glob('*HE*')])
            self.image_folders_HE.extend(image_folders_HE_current)
            image_folders_MUC_current = sorted([str(file) for file in Path(tile_folder).glob('*MUC*')])
            self.image_folders_MUC.extend(image_folders_MUC_current)
        try:
            max_patients = opt.max_patients
            print(f'use subset of {max_patients} patients')
            self.image_folders_HE = self.image_folders_HE[:max_patients]
            self.image_folders_MUC = self.image_folders_MUC[:max_patients]
        except:
            print('use all patients')
        self.A_paths = sorted([str(file) for directory in self.image_folders_HE for file in Path(directory).glob('*')])
        self.B_paths = sorted([str(file) for directory in self.image_folders_MUC for file in Path(directory).glob('*')])

        self.A_size = len(self.A_paths)  # get the size of dataset_aligned A
        print("#HE: ", self.A_size)
        self.B_size = len(self.B_paths)  # get the size of dataset_aligned B
        print("#IHC: ", self.B_size)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        try:
            if self.opt.aligned:
                index_B = index
            elif self.opt.serial_batches:  # make sure index is within then range
                index_B = index % self.B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
        except:
            if self.opt.serial_batches:  # make sure index is within then range
                index_B = index % self.B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        try:
            A_img = Image.open(A_path).convert('RGB')
            A = self.transform_A(A_img)
        except OSError as e:
            print(f"Error: Failed to open or process the image '{A_path}': {e}", file=sys.stderr)
            print("Now trying with the previous image")
            index -= 1
            if index < 0:
                index =self.A_size-1
            return self.__getitem__(index)
        try:
            B_img = Image.open(B_path).convert('RGB')
            B = self.transform_B(B_img)
        except OSError as e:
            print(f"Error: Failed to open or process the image '{B_path}': {e}", file=sys.stderr)
            print("Now trying with the previous image")
            index -= 1
            if index < 0:
                index = self.B_size-1
            return self.__getitem__(index)

        # apply image transformation
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset_aligned.

        As we have two datasets with potentially different number of images,
        we take a maximum of the two sets
        """
        return max(self.A_size, self.B_size)