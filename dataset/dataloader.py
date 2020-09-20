from bisect import bisect_right
from pathlib import Path
from typing import Iterable
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, Subset
from torchaudio.datasets import COMMONVOICE


class LanguageDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, languages: Iterable[str], num_workers=0, root=Path('./dataset/common-voice'), balanced=True):
        self.batch_size = batch_size
        self.languages = languages
        self.num_workers = num_workers
        self.root = root
        self.balanced = balanced

    def setup(self, stage=None):
        self.train_dataset = CommonVoiceDataset(
            self.root, self.languages, self.balanced, download=False, split='train'
        )
        self.validation_dataset = CommonVoiceDataset(
            self.root, self.languages, self.balanced, download=False, split='test'
        )

    def prepare_data(self):
        CommonVoiceDataset(self.root, self.languages, self.balanced, download=True, split='train')
        CommonVoiceDataset(self.root, self.languages, self.balanced, download=True, split='test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

    def validation_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True
        )

class CommonVoiceDataset(ConcatDataset):
    # if COMMONVOICE would put these in the class as I do here,
    # I wouldn't have to copy+paste it...
    supported_languages = {
            "tatar": "tt",
            "english": "en",
            "german": "de",
            "french": "fr",
            "welsh": "cy",
            "breton": "br",
            "chuvash": "cv",
            "turkish": "tr",
            "kyrgyz": "ky",
            "irish": "ga-IE",
            "kabyle": "kab",
            "catalan": "ca",
            "taiwanese": "zh-TW",
            "slovenian": "sl",
            "italian": "it",
            "dutch": "nl",
            "hakha chin": "cnh",
            "esperanto": "eo",
            "estonian": "et",
            "persian": "fa",
            "portuguese": "pt",
            "basque": "eu",
            "spanish": "es",
            "chinese": "zh-CN",
            "mongolian": "mn",
            "sakha": "sah",
            "dhivehi": "dv",
            "kinyarwanda": "rw",
            "swedish": "sv-SE",
            "russian": "ru",
            "indonesian": "id",
            "arabic": "ar",
            "tamil": "ta",
            "interlingua": "ia",
            "latvian": "lv",
            "japanese": "ja",
            "abkhaz": "ab",
            "cantonese": "zh-HK",
            "romansh sursilvan": "rm-sursilv"
        }

    def __init__(self, root, languages: Iterable[str], balanced=True, download=False, split='train'):
        assert split in ('train', 'test', 'validated')
        split += '.tsv' # don't know why torch audio commonvoice requires this ext

        for language in languages:
            assert language in self.supported_languages, (
                f"Got {language}, options are {self.supported_languages.keys()}"
            )
        
        for language in languages:
            path = f"./{root}/CommonVoice/cv-corpus-4-2019-12-10/{self.supported_languages[language]}/"
            Path(path).mkdir(parents=True, exist_ok=True)

        if download:
            try:
                datasets = [
                    COMMONVOICE(root=f"{root}/CommonVoice/cv-corpus-4-2019-12-10/{self.supported_languages[language]}/",
                    tsv=split, url=language, download=download, version='cv-corpus-4-2019-12-10')
                    for language in languages
                ]
            except FileNotFoundError:
                pass
        datasets = [
                    COMMONVOICE(root=root,
                    tsv=split, url=language, download=download, version='cv-corpus-4-2019-12-10')
                    for language in languages
                ]

        if balanced:
            min_length = min(len(dataset) for dataset in datasets)
            datasets = [
                Subset(dataset, list(*range(min_length)))
                for dataset in dataset
            ]

        super(CommonVoiceDataset, self).__init__(datasets)


    def __getitem__(self, idx):
        data = super(CommonVoiceDataset, self).__getitem__(idx)
        target = bisect_right(self.cumulative_sizes, idx)

        return data, target


if __name__ == '__main__':
    module = LanguageDataModule(languages=('votic', 'abkhaz'), batch_size=2)
    module.prepare_data()
    breakpoint()
