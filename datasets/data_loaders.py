import torch
from torchvision import transforms
from PIL import Image

from datasets.NYUDataset import NYUDataset
from datasets.NYUMultiDataset import NYUMultiDataset
from datasets.LHSynthDataset import LHSynthDataset
from datasets.LHSynthIterableDataset import LHSynthIterableDataset
from datasets.ITSEDataset import ITSEDataset
from datasets.ITSEFDataset import ITSEFDataset


class LHSynthGenDataLoader(torch.utils.data.DataLoader):
    """LibHand dynamic synthetic generator data loader."""
    def __init__(self, scene_path, pose_config_path, shape_config_path,
                 batch_per_epoch, batch_size, num_workers, num_points,
                 noise_coeff=0.0, training=True, image_size=120):
        sample_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor(),
        ])

        target_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor()
        ])

        self.dataset = LHSynthIterableDataset(scene_path, pose_config_path,
                                              shape_config_path,
                                              sample_tsfms, target_tsfms,
                                              num_points, training)
        self.batch_per_epoch = batch_per_epoch

        super(LHSynthGenDataLoader, self).__init__(self.dataset,
                                                   batch_size,
                                                   False,
                                                   num_workers=num_workers,
                                                   worker_init_fn=self.dataset.setup)

class LHSynthDataLoader(torch.utils.data.DataLoader):
    """LibHand static synthetic data loader."""
    def __init__(self, data_path, batch_size, shuffle, validation_split,
            num_workers, num_points, noise_coeff=0.0, image_size=120):
        sample_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor(),
        ])

        target_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor()
        ])

        self.data_path = data_path
        self.dataset = LHSynthDataset(data_path, sample_tsfms, target_tsfms,
                num_points, noise_coeff)
        super(LHSynthDataLoader, self).__init__(self.dataset,
                                                batch_size,
                                                shuffle,
                                                validation_split,
                                                num_workers)


class NYUDataLoader(torch.utils.data.DataLoader):
    """NYU data loader."""
    def __init__(self, data_path, batch_size, shuffle,
            num_workers, num_points, training=True):
        sample_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(120, Image.NEAREST),
            transforms.ToTensor(),
        ])

        target_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(120, Image.NEAREST),
            transforms.ToTensor()
        ])

        self.data_path = data_path
        self.dataset = NYUDataset(data_path, sample_tsfms, target_tsfms,
                [], num_points, training)
        super(NYUDataLoader, self).__init__(self.dataset,
                                            batch_size,
                                            shuffle,
                                            num_workers=num_workers)


class NYUMultiDataLoader(torch.utils.data.DataLoader):
    """NYU data loader."""
    def __init__(self, data_path, batch_size, shuffle,
            num_workers, num_points, synth=False, training=True):
        sample_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(120, Image.NEAREST),
            transforms.ToTensor(),
        ])

        target_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(120, Image.NEAREST),
            transforms.ToTensor()
        ])

        self.data_path = data_path
        self.dataset = NYUMultiDataset(data_path, sample_tsfms, target_tsfms,
                num_points, synth, training)
        super(NYUMultiDataLoader, self).__init__(self.dataset,
                                             batch_size,
                                             shuffle,
                                             num_workers=num_workers)

class ITSEDataLoader(torch.utils.data.DataLoader):
    """ITSE Data"""
    def __init__(self, data_path, batch_size, shuffle, num_workers,
                 idx_range, num_points):
        sample_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(120, Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.data_path = data_path
        self.dataset = ITSEDataset(data_path, sample_tsfms, idx_range,
                                   num_points)
        super(ITSEDataLoader, self).__init__(self.dataset,
                                             batch_size,
                                             shuffle,
                                             num_workers=num_workers)

class ITSEFDataLoader(torch.utils.data.DataLoader):
    """ITSE (Farnaz) Data"""
    def __init__(self, data_path, batch_size, shuffle, num_workers,
                 num_points):
        sample_tsfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(120, Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.data_path = data_path
        self.dataset = ITSEFDataset(data_path, sample_tsfms, num_points)
        super(ITSEFDataLoader, self).__init__(self.dataset,
                                              batch_size,
                                              shuffle,
                                              num_workers=num_workers)
