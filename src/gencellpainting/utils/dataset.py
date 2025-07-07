import os
import torchvision
import logging
import multiprocessing as mp
import torch
import tqdm


class CellPaintingDatasetInMemory(torchvision.datasets.VisionDataset):
    def __init__(self, n_images=None,paths=None, tensor=None, **kwargs):
        super().__init__(**kwargs)
        # Initializing the dataset
        self.paths = None

        self.height = None
        self.width = None
        self.n_channels = None
        self.n_samples = None
        self.n_images = n_images

        self.tensor = None
        if tensor is  None:
            self.paths = paths
            if self.paths is None:
                self._build_paths()
            self._load_data()
        else:
            self.tensor = tensor
        self._compute_metadatas()

    def _build_paths(self):
        self.paths = [x for x in os.listdir(self.root)]
        self.paths = self.paths[:self.n_images]
        logging.info(f"Found {len(self.paths)} paths")

    def _load_data(self):
        logging.info(f"Loading data from {self.root}")

        # Load data without multiprocessing as it crashes
        tensors = [torch.load(os.path.join(self.root, x)) for x in tqdm.tqdm(self.paths)]
        
        self.tensor = torch.stack(tensors)

    def _compute_metadatas(self):
        N, C, H, W = self.tensor.shape
        self.height = H
        self.width = W
        self.n_channels = C
        self.n_samples = N
    
    def __getitem__(self, index):
        img = self.tensor[index]
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return self.n_samples
    

class WGANCriticDataset(torchvision.datasets.VisionDataset):
    def __init__(self, dataset, ncritic=5):
        self.dataset = dataset
        self.ncritic = ncritic
        self.n_samples = dataset.n_samples

    def __getitem__(self, index):
        # Random samping of ncritic intex
        random_idx = torch.randint(0, self.n_samples, (self.ncritic,))
        items = [self.dataset[i] for i in random_idx]
        return self.dataset[index],items
    
    def __len__(self):
        return self.n_samples
