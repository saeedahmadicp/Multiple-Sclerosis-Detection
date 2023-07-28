from torch.utils.data import DataLoader

from .dataset import MultipleSclerosisDataset

__all__ = ['get_train_ds_loader', 'get_test_ds_loader']

def get_train_ds_loader(dataset_dir, data_dict, batch_size, modality="FLAIR", transform=None, target_transform=None):
    ds = MultipleSclerosisDataset(dataset_dir, data_dict, 'train', modality, transform, target_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)


def get_test_ds_loader(dataset_dir, data_dict, batch_size, modality="FLAIR", transform=None, target_transform=None):
    ds = MultipleSclerosisDataset(dataset_dir, data_dict, 'test', modality, transform, target_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)