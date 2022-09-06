from .datamodule import DataModule


def build_datamodule(opt, is_train):
    return DataModule(opt, is_train=is_train)

