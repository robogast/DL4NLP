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
                        # layers=14,#10
                        #  blocks=1,#4
                        skip_channels=32, #256 
                        end_channels=32, #256
                        # uncomment for fast debug network
                    )

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['state_dict'])


    trainer = pl.Trainer(

        # comment to run on cpu for local testing
        gpus=args.gpus,
        auto_select_gpus=True,
        # distributed_backend='ddp',
        benchmark=True,
        ## -------

        terminate_on_nan=True,
    )
    datamodule.setup()

    # trainer.fit(model, datamodule)


    results = trainer.test(model, datamodule.test_dataloader())

    # print(results)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--dataset-path", type=Path, default=Path())
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--languages", nargs="+", help='Languages to use for model training',
                        required=True, choices=tuple(CommonVoiceDataset.supported_languages.keys()))
    parser.add_argument("--num-workers", help='Num workers to use (per gpu!)', type=int, default=0)
    parser.add_argument("--ckpt_path", default="/home/lgpu0202/DL4NLP/epoch2.ckpt", type=str)
    args = parser.parse_args()

    main(args)