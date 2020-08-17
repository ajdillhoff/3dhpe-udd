import json
import argparse
from argparse import Namespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.DepthGenLM import DepthGenLM


def main(args):
    torch.autograd.set_detect_anomaly(True)
    hparams = Namespace(**json.load(open((args.config))))

    model = DepthGenLM(hparams)
    if args.resume is not None:
        pretrain = torch.load(args.resume)
        model.load_state_dict(pretrain['state_dict'])

    checkpoint_callback = ModelCheckpoint(filepath=hparams.save_path,
                                          monitor='val_loss', mode='min',
                                          period=0, verbose=1)
    early_stop_callback = EarlyStopping('val_loss', patience=hparams.patience,
                                        verbose=1)
    trainer = Trainer(val_check_interval=hparams.val_check_interval,
                      #check_val_every_n_epoch=hparams.val_check_interval,
                      gpus=hparams.gpus,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=None,
                      progress_bar_refresh_rate=1,
                      num_training_batches=10)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='checkpoint to resume from (default: None)')
    args = parser.parse_args()
    main(args)
