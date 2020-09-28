from bisect import bisect_right
from pathlib import Path
from typing import Iterable
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, Subset
from torchaudio.datasets import COMMONVOICE
import numpy as np

class LanguageDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, languages: Iterable[str], num_workers=0, root=Path(), balanced=True):
        super(LanguageDataModule, self).__init__()
        self.batch_size = batch_size
        self.languages = languages
        self.num_workers = num_workers
        self.root = root
        self.balanced = balanced


    def setup(self, stage=None):
        self.train_dataset = CommonVoiceDataset(
            self.root, self.languages, self.balanced, split='train'
        )
        self.validation_dataset = CommonVoiceDataset(
            self.root, self.languages, self.balanced, split='test'
        )

    def prepare_data(self):
        CommonVoiceDataset.download(self.root, self.languages)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        # FIXME: change val dataloader to get val data from train dataset instead of test dataset
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
    version = 'cv-corpus-4-2019-12-10'
    dataset_dir = 'CommonVoice'


    def __init__(self, root: Path, languages: Iterable[str], balanced=True, split='train'):
        assert split in ('train', 'test', 'validated')
        split += '.tsv'

        self.root = root
        self.languages = languages

        self._item_length = 100000
        self.classes = 256

        for language in languages:
            assert language in self.supported_languages, (
                f"Got {language}, options are {self.supported_languages.keys()}"
            )

        # FIXME: add appropriate message if dataset isn't downloaded first
        datasets = [
            COMMONVOICE(root=root,
            tsv=split, url=language, download=False, version=self.version)
            for language in languages
        ]
        if balanced:
            min_length = min(len(dataset) for dataset in datasets)
            datasets = [
                Subset(dataset, list(range(min_length)))
                for dataset in datasets
            ]
        super(CommonVoiceDataset, self).__init__(datasets)


    def __getitem__(self, idx):
        data = super(CommonVoiceDataset, self).__getitem__(idx)[0]
        quantized = torch.from_numpy(quantize_data(data.squeeze(0), self.classes))

        one_hot = torch.FloatTensor(self.classes, self._item_length).zero_()
        one_hot.scatter_(0, quantized[:self._item_length].unsqueeze(0), 1.)

        target = bisect_right(self.cumulative_sizes, idx)

        return one_hot, target
        
    @classmethod
    def _get_language_dir(cls, root, language):
        return root / cls.dataset_dir / cls.version / cls.supported_languages[language]

    @classmethod
    def download(cls, root, languages):
        for language in languages:
            language_path = cls._get_language_dir(root, language)
            if not language_path.exists():
                language_path.mkdir(parents=True, exist_ok=True)
                try:
                    print('downloaden')
                    COMMONVOICE(
                        root=language_path,
                        tsv='', url=language, download=True, version=cls.version
                    )
                except FileNotFoundError:
                    pass


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized

def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s