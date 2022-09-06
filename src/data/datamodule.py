from importlib import import_module

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .base_dataset import BaseDataset

class DataModule(LightningDataModule):
    def __init__(self, opt, is_train=False):
        super().__init__()
        self.opt = opt
        self.is_train = is_train

        self.trainset_type = opt['train']['type'] if is_train else None
        self.validset_type = opt['valid']['type'] if is_train else None
        self.testset_type = opt['test']['type']

    def setup(self, stage=None):
        datasets = {
            'BaseDataset' : BaseDataset
        }
        try:
            self.train_dataset = datasets[self.trainset_type](self.opt['train']) if self.is_train else None
            self.valid_dataset = datasets[self.validset_type](self.opt['valid']) if self.is_train else None
            self.test_dataset = datasets[self.testset_type](self.opt['test'])
        except:
            raise NotImplementedError(f'Dataset not implemented: {self.trainset_type}, {self.validset_type}, {self.testset_type}')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.opt['train']['batch_size_per_gpu'], 
            num_workers=self.opt['train']['num_workers'], 
            pin_memory=self.opt['train']['pin_memory'],
            drop_last=True,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.opt['valid']['batch_size_per_gpu'], 
            num_workers=self.opt['valid']['num_workers'],
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.opt['test']['batch_size_per_gpu'], 
            num_workers=self.opt['test']['num_workers'],
            shuffle=False)