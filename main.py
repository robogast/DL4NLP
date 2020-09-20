from dataset.dataloader import LanguageDataModule
from litclassifier import LitClassifier


if __name__ == '__main__':
    module = LanguageDataModule(languages=('abkhaz', 'esperanto'), batch_size=2)
    module.prepare_data()

    # model
    model = LitClassifier()

    # training
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=2, batch_size=2, limit_train_batches=200)
    trainer.fit(model, module)

    trainer.test(test_dataloaders=test_dataset)


    breakpoint()


