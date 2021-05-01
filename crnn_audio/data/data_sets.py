"""
Dataset in charge of knowing where the source of data, labels and transforms.
Should provide access to the data by indexing.
"""

import torch.utils.data as data


class FolderDataset(data.Dataset):

    def __init__(self, data_arr, load_func, transforms=None):
        self.transforms = transforms
        self.data_arr = data_arr
        self.load_func = load_func

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, index):
        elem = self.data_arr[index]
        data_t, label = self.load_func(elem['path']), elem['class_idx']

        if self.transforms is not None:
            audio, sr, label = self.transforms.apply(data_t, label)
            return audio, sr, label

        return data_t, label


if __name__ == '__main__':
    pass
