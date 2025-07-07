import os
import numpy

from typing import *

from .dataset import CellPaintingDatasetInMemory
from torch.utils.data import DataLoader, random_split
from torch import Generator

def create_datasets(images_folder:str,transforms,n_train:Union[int,float],\
                    nmax: Optional[int]=None, random_seed:int=42):
    
    # Listing all images
    paths = [x for x in os.listdir(images_folder) if x.endswith(".pt")]

    # Sampling correct paths number
    numpy.random.seed(random_seed)
    if nmax is not None:
        paths = numpy.random.choice(paths,size=nmax,replace=False)

    train_size = n_train
    if n_train <= 1.:
        train_size = int(len(paths)*n_train)
    test_size = len(paths) - train_size
    
    random_generator = Generator().manual_seed(10 * random_seed)
    paths_train, paths_test = random_split(paths, [train_size, test_size],)

    ds_train = CellPaintingDatasetInMemory(root=images_folder,transform=transforms, paths=paths_train)
    ds_test = CellPaintingDatasetInMemory(root=images_folder,transform=transforms, paths=paths_test)

    return ds_train, ds_test

def create_dataloaders(images_folder:str,transforms,n_train:Union[int,float]=0.8,\
                    nmax: Optional[int]=None, random_seed:int=42, batch_size:int=64,\
                    shuffle=True, num_workers=4, **kwargs):
    ds_train, ds_test = create_datasets(images_folder,transforms,n_train,nmax,random_seed)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
    return dl_train, dl_test, ds_train, ds_test