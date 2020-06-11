from torch.utils.data import DataLoader

from DataAugmentation import get_transforms
from TrainCsvReader import TrainCsvReader
from WheatDataset import WheatDataset


class WheatDataLoader:
    def get_data_loaders(self, dir_input, dir_train, debug=False):
        trainCsvReader = TrainCsvReader(dir_input=dir_input)
        df = trainCsvReader.preprocess()
        train_df, valid_df = trainCsvReader.train_test_split(df=df, test_frac=.3)

        if debug:
            train_batch_size = 1
            validation_batch_size = 1
        else:
            train_batch_size = 16
            validation_batch_size = 8

        train_dataset = WheatDataset(dataframe=train_df, image_dir=dir_train, transforms=get_transforms(type='train'))
        valid_dataset = WheatDataset(dataframe=valid_df, image_dir=dir_train,
                                     transforms=get_transforms(type='validation'))
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )

        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
        )

        return train_data_loader, valid_data_loader

    @staticmethod
    def _collate_fn(batch):
        return tuple(zip(*batch))
