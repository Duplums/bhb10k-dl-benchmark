from torch.utils.data import Dataset
import pandas as pd
import torch
from data.augmentation import *

class MRIDataset(Dataset):

    def __init__(self, config, args, training=False, validation=False, test=False, **kwargs):
        super().__init__(**kwargs)
        assert training + validation + test == 1

        self.transforms = Transformer()
        self.config = config
        self.args = args
        # Crop+Pad images to have fixed dimension (1, 128, 128, 128)
        self.transforms.register(crop, probability=1.0, size=(1, 121, 128, 121))
        self.transforms.register(padding, probability=1.0, size=(1, 128, 128, 128))
        self.transforms.register(normalize, probability=1.0)
        if (not validation) and (not test):
            self.add_data_augmentations(self.transforms, args.da)
        if training:
            self.data = np.load(args.train_data_path)
            self.labels = pd.read_csv(args.train_label_path)
        elif validation:
            self.data = np.load(args.val_data_path)
            self.labels = pd.read_csv(args.val_label_path)
        elif test:
            self.data = np.load(args.test_data_path)
            self.labels = pd.read_csv(args.test_label_path)


    def add_data_augmentations(self, transformer, augmentations):
        aug2tf = {
            'flip': (flip, dict()),
            'blur': (add_blur, {'sigma': [0.1, 1]}),
            'noise': (add_noise, {'sigma': [0.1, 1]}),
            'resized_crop': (crop, {'size': (1, 90, 90, 90), 'crop_type': 'random', 'resize':True}),
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
        x = self.transforms(self.data[idx])
        labels = self.labels[self.args.labels].values[idx]

        return (x, labels)

    def __len__(self):
        return len(self.data)