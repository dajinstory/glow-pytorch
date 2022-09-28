import os
import yaml
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data import build_datamodule
from model import build_model


# Parser
## Config Parser
config_parser = argparse.ArgumentParser(description='Config file path')
config_parser.add_argument('-c', '--config', default='config/glow_v0.yml', metavar='FILE')
args_config, remaining = config_parser.parse_known_args()

## Additional Argument Parser
parser = argparse.ArgumentParser(description='Config file to arguments')
parser.add_argument('--seed', type=int, default=310)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--ckpt_path', type=str, default='')

## Parse arguments
with open(args_config.config, 'r') as f:
    cfg = yaml.safe_load(f)
    parser.set_defaults(**cfg)
args = parser.parse_args(remaining)


# Log, Checkpoint
experiment_path = '/data/dajinhan/experiment' # './experiment'
args.save_path = os.path.join(experiment_path, args.name, 'checkpoint')
args.log_path = os.path.join(experiment_path, args.name, 'log')
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.log_path, exist_ok=True)
logger = TensorBoardLogger(args.log_path, name='flow')


# Random seed
seed_everything(args.seed)


# Datamodule
datamodule = build_datamodule(args.DATA, is_train=True)


# Model
model = build_model(args.MODEL, is_train=True)
ckpt_path = args.ckpt_path if args.resume else None


# Callbacks
## Callback - Checkpoint
checkpoint_callback_nll = ModelCheckpoint(
    dirpath=args.save_path,
    filename='best-epoch{epoch:02d}-nll{val/metric/nll:.4f}',
    monitor='val/metric/nll',
    save_last=True,
    save_top_k=3,
    mode='min',
    auto_insert_metric_name=False)

checkpoint_callbacks = [
    checkpoint_callback_nll]

## Callback - LR monitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')

## Total Callbacks
total_callbacks = [*checkpoint_callbacks, lr_monitor]


# Train
## Lightning Trainer
trainer = Trainer(
    strategy='ddp', # 'ddp' #for python
    accelerator='gpu', 
    devices=args.gpus,
    logger=logger,
    callbacks=total_callbacks,
    default_root_dir=args.save_path,
    max_epochs=args.n_epochs, 
    val_check_interval=args.val_check_interval,
    log_every_n_steps=args.log_every_n_steps,
    resume_from_checkpoint=ckpt_path)
## Train model
trainer.fit(model=model, datamodule=datamodule)
