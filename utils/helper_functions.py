import random

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from data.shift_dataset import ShiftDataset, LabelFilter, shift_label_mapping
from data.streethazards_dataset import StreetHazardsDataset, streethazards_label_mapping
from data.coco_dataset import COCODataset, USED_CATEGORIES
from data.dataset_handler import RandomCropFlipDataset, OutlierDataset

class ShiftSegmentationDataModule(LightningDataModule):

    def __init__(self,
            dataset_dir: str,
            training_size: int,
            batch_size: int,
            label_filter: LabelFilter = None,
            test_label_filter: LabelFilter = None,
            label_mapping: str = "normal",
            num_workers: int = 8,
            val_amount: float = .05) -> None:
        '''
            Loads the Shift Dataset (train and val splits)

            The train split is randomly split again
        '''
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        shift_train = ShiftDataset(
            dataset_dir, "train",
            label_mapping=shift_label_mapping[label_mapping],
            label_filter=label_filter)
        self.shift_val = ShiftDataset( # shift_val will be empty
            dataset_dir, "train",
            label_mapping=shift_label_mapping[label_mapping],
            label_filter=LabelFilter("0", 1e10, 1e11))

        random.seed(42)
        random.shuffle(shift_train.files)
        train_num_images = int((1-val_amount)*len(shift_train.files))
        train_images = shift_train.files[:train_num_images]
        val_images = shift_train.files[train_num_images:]

        shift_train.files = train_images
        self.shift_val.files = val_images

        # we might want to load the test dataset with other kind of filtering of labels
        # the test
        self.shift_test = ShiftDataset(
            dataset_dir, "val",
            label_mapping=shift_label_mapping[label_mapping], label_filter=test_label_filter)

        self.train_dataset = RandomCropFlipDataset(shift_train, training_size, scale=(.65, 1.5))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.shift_val, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.shift_test, self.batch_size, shuffle=False, num_workers=self.num_workers)


class ShiftOODDataModule(LightningDataModule):

    def __init__(self,
            dataset_dir: str,
            training_size: int,
            outlier_dir: str,
            outlier_max_size: int,
            batch_size: int,
            label_filter: LabelFilter = None,
            test_label_filter: LabelFilter = None,
            label_mapping: str = "normal",
            horizon: float = 0,
            alpha_blend: float = 1,
            histogram_matching: bool = False,
            blur: int = 0,
            num_workers: int = 8,
            val_amount: float = .05
        ) -> None:
        '''
            Loads the Shift Dataset (train and val splits)

            The train split is randomly split again
        '''
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        # load the outlier dataset
        coco_dataset = COCODataset(
            outlier_dir, outlier_max_size, USED_CATEGORIES
        )

        shift_train = ShiftDataset(
            dataset_dir, "train",
            label_mapping=shift_label_mapping[label_mapping], label_filter=label_filter)
        shift_val = ShiftDataset( # shift_val will be empty
            dataset_dir, "train",
            label_mapping=shift_label_mapping[label_mapping],
            label_filter=LabelFilter("0", 1e10, 1e11))

        random.shuffle(shift_train.files)
        train_num_images = int((1-val_amount)*len(shift_train.files))
        train_images = shift_train.files[:train_num_images]
        val_images = shift_train.files[train_num_images:]

        shift_train.files = train_images
        shift_val.files = val_images

        # we might want to load the test dataset with other kind of filtering of labels
        # the test
        self.shift_test = ShiftDataset(
            dataset_dir, "val",
            label_mapping=shift_label_mapping[label_mapping], label_filter=test_label_filter)

        self.train_dataset = OutlierDataset(
            shift_train, coco_dataset, training_size, scale=(.65, 1.5),
            horizon=horizon, alpha_blend=alpha_blend,
            histogram_matching=histogram_matching, blur=blur
        )
        self.val_dataset = OutlierDataset(
            shift_val, coco_dataset, training_size, scale=(.65, 1.5),
            horizon=horizon, alpha_blend=alpha_blend,
            histogram_matching=histogram_matching, blur=blur
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.shift_test, self.batch_size, shuffle=False, num_workers=self.num_workers)


class StreetHazardsDataModule(LightningDataModule):

    def __init__(self,
            dataset_dir: str,
            training_size: int,
            batch_size: int,
            label_mapping: str = "normal",
            num_workers: int = 8
        ) -> None:
        '''
            Loads the StreetHazards Dataset (train, val and test splits)
        '''
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        streethazards_train = StreetHazardsDataset(
            dataset_dir, "train", label_mapping=streethazards_label_mapping[label_mapping])
        self.streethazards_val = StreetHazardsDataset(
            dataset_dir, "val", label_mapping=streethazards_label_mapping[label_mapping])
        self.streethazards_test = StreetHazardsDataset(
            dataset_dir, "test", label_mapping=streethazards_label_mapping[label_mapping])

        self.train_dataset = RandomCropFlipDataset(
            streethazards_train, training_size, scale=(.75, 1.5))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.streethazards_val, self.batch_size,
            shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.streethazards_test, self.batch_size,
            shuffle=False, num_workers=self.num_workers)


class StreetHazardsOODDataModule(LightningDataModule):

    def __init__(self,
            dataset_dir: str,
            training_size: int,
            outlier_dir: str,
            outlier_max_size: int,
            batch_size: int,
            label_mapping: str = "normal",
            horizon: float = 0,
            alpha_blend: float = 1,
            histogram_matching: bool = False,
            blur: int = 0,
            num_workers: int = 8
        ) -> None:
        '''
            Loads the Shift Dataset (train and val splits)

            The train split is randomly split again
        '''
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        # load the outlier dataset
        coco_dataset = COCODataset(
            outlier_dir, outlier_max_size, USED_CATEGORIES
        )

        streethazards_train = StreetHazardsDataset(
            dataset_dir, "train", label_mapping=streethazards_label_mapping[label_mapping])
        streethazards_val = StreetHazardsDataset(
            dataset_dir, "val", label_mapping=streethazards_label_mapping[label_mapping])
        self.streethazards_test = StreetHazardsDataset(
            dataset_dir, "test", label_mapping=streethazards_label_mapping[label_mapping])

        self.train_dataset = OutlierDataset(
            streethazards_train, coco_dataset, training_size, scale=(.75, 1.5),
            horizon=horizon, alpha_blend=alpha_blend,
            histogram_matching=histogram_matching, blur=blur
        )
        self.val_dataset = OutlierDataset(
            streethazards_val, coco_dataset, training_size, scale=(.75, 1.5),
            horizon=horizon, alpha_blend=alpha_blend,
            histogram_matching=histogram_matching, blur=blur
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, self.batch_size,
            shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.streethazards_test, self.batch_size,
            shuffle=False, num_workers=self.num_workers)
