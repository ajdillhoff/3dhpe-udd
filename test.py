import json
import argparse
from argparse import Namespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model.DepthGenLM import DepthGenLM


def main(args):
    if args.config is None:
        model = DepthGenLM.load_from_checkpoint(args.checkpoint)
    else:
        hparams = Namespace(**json.load(open((args.config))))
        model = DepthGenLM(hparams)
        pretrain = torch.load(args.checkpoint)
        model.load_state_dict(pretrain['state_dict'])

    trainer = Trainer(val_check_interval=100, gpus=1)
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', default=None)
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
    main(args)

