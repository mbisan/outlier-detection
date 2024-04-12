import random

from torch.utils.data import DataLoader

from data.shift_dataset import ShiftDataset, LabelFilter, shift_label_mapping
from data.streethazards_dataset import StreetHazardsDataset, streethazards_label_mapping
from data.coco_dataset import COCODataset, USED_CATEGORIES
from data.dataset_handler import RandomCropFlipDataset, OutlierDataset

def load_shift_segmentation(
        dataset_dir: str,
        training_size: int,
        batch_size: int,
        label_filter: LabelFilter = None,
        test_label_filter: LabelFilter = None,
        label_mapping: str = "normal",
        num_workers: int = 8,
        val_amount: float = .05
    ):
    '''
        Loads the Shift Dataset (train and val splits)

        The train split is randomly split again
    '''

    shift_train = ShiftDataset(
        dataset_dir, "train",
        label_mapping=shift_label_mapping[label_mapping], label_filter=label_filter)
    shift_val = ShiftDataset( # shift_val will be empty
        dataset_dir, "train",
        label_mapping=shift_label_mapping[label_mapping], label_filter=LabelFilter("0", 1e10, 1e11))

    random.shuffle(shift_train.files)
    train_num_images = int((1-val_amount)*len(shift_train.files))
    train_images = shift_train.files[:train_num_images]
    val_images = shift_train.files[train_num_images:]

    shift_train.files = train_images
    shift_val.files = val_images

    # we might want to load the test dataset with other kind of filtering of labels
    # the test
    shift_test = ShiftDataset(
        dataset_dir, "val",
        label_mapping=shift_label_mapping[label_mapping], label_filter=test_label_filter)

    train_dataset = RandomCropFlipDataset(shift_train, training_size, scale=(.65, 1.5))
    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(shift_val, batch_size//4, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(shift_test, batch_size//4, shuffle=False, num_workers=num_workers)

    return (train_dataloader, val_dataloader, test_dataloader)

def load_shift_ood(
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
    ):
    '''
        Loads the Shift Dataset (train and val splits)

        The train split is randomly split again
    '''

    # load the outlier dataset
    coco_dataset = COCODataset(
        outlier_dir, outlier_max_size, USED_CATEGORIES
    )

    shift_train = ShiftDataset(
        dataset_dir, "train",
        label_mapping=shift_label_mapping[label_mapping], label_filter=label_filter)
    shift_val = ShiftDataset( # shift_val will be empty
        dataset_dir, "train",
        label_mapping=shift_label_mapping[label_mapping], label_filter=LabelFilter("0", 1e10, 1e11))

    random.shuffle(shift_train.files)
    train_num_images = int((1-val_amount)*len(shift_train.files))
    train_images = shift_train.files[:train_num_images]
    val_images = shift_train.files[train_num_images:]

    shift_train.files = train_images
    shift_val.files = val_images

    # we might want to load the test dataset with other kind of filtering of labels
    # the test
    shift_test = ShiftDataset(
        dataset_dir, "val",
        label_mapping=shift_label_mapping[label_mapping], label_filter=test_label_filter)

    train_dataset = OutlierDataset(
        shift_train, coco_dataset, training_size, scale=(.65, 1.5),
        horizon=horizon, alpha_blend=alpha_blend, histogram_matching=histogram_matching, blur=blur
    )
    val_dataset = OutlierDataset(
        shift_val, coco_dataset, training_size, scale=(.65, 1.5),
        horizon=horizon, alpha_blend=alpha_blend, histogram_matching=histogram_matching, blur=blur
    )

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size//4, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(shift_test, batch_size//4, shuffle=False, num_workers=num_workers)

    return (train_dataloader, val_dataloader, test_dataloader)


def load_streethazards_segmentation(
        dataset_dir: str,
        training_size: int,
        batch_size: int,
        label_mapping: str = "normal",
        num_workers: int = 8,
    ):
    '''
        Loads the StreetHazards Dataset (train, val and test splits)
    '''

    streethazards_train = StreetHazardsDataset(
        dataset_dir, "train", label_mapping=streethazards_label_mapping[label_mapping])
    streethazards_val = StreetHazardsDataset(
        dataset_dir, "val", label_mapping=streethazards_label_mapping[label_mapping])
    streethazards_test = StreetHazardsDataset(
        dataset_dir, "test", label_mapping=streethazards_label_mapping[label_mapping])

    train_dataset = RandomCropFlipDataset(streethazards_train, training_size, scale=(.75, 1.5))
    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(
        streethazards_val, batch_size//4, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(
        streethazards_test, batch_size//4, shuffle=False, num_workers=num_workers)

    return (train_dataloader, val_dataloader, test_dataloader)

def load_streethazards_ood(
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
        num_workers: int = 8,
    ):
    '''
        Loads the Shift Dataset (train and val splits)

        The train split is randomly split again
    '''

    # load the outlier dataset
    coco_dataset = COCODataset(
        outlier_dir, outlier_max_size, USED_CATEGORIES
    )

    streethazards_train = StreetHazardsDataset(
        dataset_dir, "train", label_mapping=streethazards_label_mapping[label_mapping])
    streethazards_val = StreetHazardsDataset(
        dataset_dir, "val", label_mapping=streethazards_label_mapping[label_mapping])
    streethazards_test = StreetHazardsDataset(
        dataset_dir, "test", label_mapping=streethazards_label_mapping[label_mapping])

    train_dataset = OutlierDataset(
        streethazards_train, coco_dataset, training_size, scale=(.75, 1.5),
        horizon=horizon, alpha_blend=alpha_blend, histogram_matching=histogram_matching, blur=blur
    )
    val_dataset = OutlierDataset(
        streethazards_val, coco_dataset, training_size, scale=(.75, 1.5),
        horizon=horizon, alpha_blend=alpha_blend, histogram_matching=histogram_matching, blur=blur
    )

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size//4, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(
        streethazards_test, batch_size//4, shuffle=False, num_workers=num_workers)

    return (train_dataloader, val_dataloader, test_dataloader)
