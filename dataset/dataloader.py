from tarfile import TarFile
from collections import namedtuple
from pathlib import Path
from subprocess import run

import torch
import pytorch_lightning as pl


class LanguageDataModule(pl.LightningDataModule):

    Language = namedtuple("Language", ("name", "code", "url"))

    languages = {
        Language(name="Dutch",      url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/nl.tar.gz",      code="nl"),
        Language(name="English",    url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz",      code="en"),
        Language(name="German",     url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/de.tar.gz",      code="de"),
        Language(name="Frysian",    url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/fy-NL.tar.gz",   code="fy-NL"),
        Language(name="French",     url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/fr.tar.gz",      code="fr"),
        Language(name="Spanish",    url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/es.tar.gz",      code="es"),
        Language(name="Welsh",      url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/cy.tar.gz",      code="cy"),
        Language(name="Russian",    url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/ru.tar.gz",      code="ru"),
        Language(name="Portuguese", url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/pt.tar.gz",      code="pt"),
        Language(name="Taiwanese",  url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/zh-TW.tar.gz",   code="zh-TW"),
        Language(name="Polish",     url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/pl.tar.gz",      code="pl"),
        Language(name="Chinese",    url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/zh-CN.tar",      code="zh-CN"),
    }

    def __init__(self, batch_size, num_workers=0, root=Path(), root_postfix='common-voice', download=True):
        self.dataset_dir = root / root_postfix
        self.download = download

    def setup(self, stage=None):
        pass

    def prepare_data(self):
        '''TODO asyncio this'''
        if self.download:
            for language in self.languages:
                run(['wget', "--continue", "--directory-prefix", f"{self.dataset_dir}", f"{language.url}"], check=True)

        for language in self.languages:
            TarFile.extractall(self.dataset_dir / f"{language.code}.tar.gz")


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

if __name__ == '__main__':
    module = LanguageDataModule(batch_size=2)
    module.prepare_data()
    breakpoint()