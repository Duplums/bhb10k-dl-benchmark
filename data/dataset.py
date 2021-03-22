from torch.utils.data import Dataset
import pandas as pd
import torch
from data.augmentation import *

class MRIDataset(Dataset):

    def __init__(self, config, args, training=False, validation=False, **kwargs):
        super().__init__(**kwargs)
        assert training != validation

        self.transforms = Transformer()
        self.config = config
        self.args = args
        self.transforms.register(normalize, probability=1.0)
        self.add_data_augmentations(self.transforms, args.da)

        if training:
            self.data = np.load(config.data_train)
            self.labels = pd.read_csv(config.label_train)
        elif validation:
            self.data = np.load(config.data_val)
            self.labels = pd.read_csv(config.label_val)

    def add_data_augmentations(self, transformer, augmentations):
        aug2tf = {
            'flip': (flip, dict()),
            'blur': (add_blur, {'snr': 1000}),
            'noise': (add_noise, {'snr': 1000}),
            'resized_crop': (crop, {'size': (115, 138, 115), 'resize':True}),
            'affine': (affine, {'rotation': 5, 'translation': 10, 'zoom': 0}),
            'ghosting': (add_ghosting, {'intensity': 1, 'axis': 0}),
            'motion': (add_motion, {'n_transforms': 3, 'rotation': 40, 'translation': 10}),
            'spike': (add_spike, {'n_spikes': 10, 'intensity': 1}),
            'biasfield': (add_biasfield, {'coefficients': 0.7}),
            'swap': (add_swap, {'num_iterations': 20}),
        }
        if augmentations is not None:
            for aug in augmentations:
                transformer.register(aug2tf[aug][0], probability=0.5, **aug2tf[aug][1])

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)
        return (list_x, list_y)

    def __getitem__(self, idx):
        np.random.seed()
        x = self.transforms(self.data[idx])
        labels = self.labels[self.args.labels].values[idx]

        return (x, labels)

    def __len__(self):
        return len(self.data)