from argparse import ArgumentParser
from pathlib import Path

import torch
import pytorch_lightning as pl

from model import LanguageModel
from dataset.dataloader import LanguageDataModule, CommonVoiceDataset

def main(args):
    datamodule = LanguageDataModule(
        root=args.dataset_path,
        languages=args.languages,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = LanguageModel(
                 skip_channels=32,
                 end_channels=32,
    )


    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(save_last=True, save_top_k=3)

    trainer = pl.Trainer(

        # comment to run on cpu for local testing
        gpus=args.gpus,
        auto_select_gpus=True,
        distributed_backend='ddp',
        benchmark=True,

        terminate_on_nan=True,

        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--dataset-path", type=Path, default=Path())
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--languages", nargs="+", help='Languages to use for model training',
                        required=True, choices=tuple(CommonVoiceDataset.supported_languages.keys()))
    parser.add_argument("--num-workers", help='Num workers to use (per gpu!)', type=int, default=6)
    args = parser.parse_args()

    main(args)