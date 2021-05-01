import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import FEATURES, ALL_FEATURES_COARSE, ALL_FEATURES_FINE, ALL_FEATURES_NO


def import_data(path, segmentation_type, drop_user_features=False, return_type='pd', drop_expert=True, is_test=False):
    """
    Import data
    :param path: path of data
    :type path: str
    :param segmentation_type: 'no', 'coarse', or 'fine'
    :type segmentation_type: str
    :param drop_user_features: specify if user features should be dropped
    :type drop_user_features: bool
    :param return_type: 'pd', 'np'
    :param drop_expert: specify if expert column should be dropped
    :type drop_expert: bool
    :param is_test: specify if expert column should be dropped
    :type is_test: bool
    :type return_type: str
    :return: dataframes containing features and labels
    """

    if segmentation_type not in ["fine", "coarse", "no"]:
        raise Exception

    if return_type not in ['pd', 'np']:
        raise Exception

    if is_test:
        if segmentation_type == 'coarse':
            df_features = pd.read_csv(f'{path}/test/features_test_{segmentation_type}_segmentation.csv',
                                      index_col=False, names=ALL_FEATURES_COARSE)
        if segmentation_type == 'fine':
            df_features = pd.read_csv(f'{path}/test/features_test_{segmentation_type}_segmentation.csv',
                                      index_col=False, names=ALL_FEATURES_FINE)
        if segmentation_type == 'no':
            df_features = pd.read_csv(f'{path}/test/features_test_{segmentation_type}_segmentation.csv',
                                      index_col=False, names=ALL_FEATURES_NO)

    else:
        df_features = pd.read_csv(f'{path}/features_{segmentation_type}_segmentation.csv', index_col=0)
        df_labels = pd.read_csv(f'{path}/labels_{segmentation_type}_segmentation.csv', index_col=0)

        if segmentation_type in ('fine', 'coarse'):
            df_features = create_multi_index(df_features)
            df_labels = create_multi_index(df_labels)
        else:
            df_features["subject"] = df_features["File_Name"]
            df_features.set_index("subject", inplace=True)
            df_features = df_features.drop(["File_Name"], axis=1)
            df_labels["subject"] = df_labels["File_Name"]
            df_labels.set_index("subject", inplace=True)
            df_labels = df_labels.drop(["File_Name"], axis=1)

    if drop_expert:
        df_features.drop(['Expert'], axis=1, errors='ignore', inplace=True)

    if drop_user_features:
        df_features.drop(FEATURES['METADATA'], axis=1, errors='ignore', inplace=True)

    if return_type == 'pd':
        if is_test:
            return df_features
        return df_features, df_labels

    if is_test:
        return df_features.values, list(df_features.columns)

    subject_indices = get_subjects_indices(df_features.index.get_level_values('subject'))

    return df_features.values, df_labels.values, subject_indices, list(df_features.columns)


def create_multi_index(data):
    data["subject"] = data["File_Name"].apply(lambda r: r.split("_")[0])
    data["file_id"] = data["File_Name"].apply(lambda r: r.split("_")[1])
    data.set_index(['subject', 'file_id'], inplace=True)
    data.drop(["File_Name"], axis=1, errors='ignore', inplace=True)

    return data


def split_experts(X, y):
    """
    Import data
    :param X: training data
    :type X: pd.DataFrame
    :param y: labels
    :type y: pd.DataFrame
    :return: split data and labels for each expert
    """
    merged = X.merge(y, left_index=True, right_index=True)

    X_exp_1 = merged[merged['Expert'] == 1].iloc[:, :-1].drop(columns=['Expert'], axis=1)
    y_exp_1 = merged[merged['Expert'] == 1].iloc[:, -1]

    X_exp_2 = merged[merged['Expert'] == 2].iloc[:, :-1].drop(columns=['Expert'], axis=1)
    y_exp_2 = merged[merged['Expert'] == 2].iloc[:, -1]

    X_exp_3 = merged[merged['Expert'] == 3].iloc[:, :-1].drop(columns=['Expert'], axis=1)
    y_exp_3 = merged[merged['Expert'] == 3].iloc[:, -1]

    return X_exp_1, y_exp_1, X_exp_2, y_exp_2, X_exp_3, y_exp_3


class CoughDataset(Dataset):
    """
    Custom torch dataset, used in order to make use of torch data loaders.
    """

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def get_subjects_indices(subject_names):
    """
    Given a list of subject names, this function will assign numeric values,
    corresponding to the group of each data sample.

    Args:
        subject_names (list): list of file names
    Returns:
        (list): list of group values
    """
    # get a unique list of names
    unique_subject_names_dict = {s: index for index, s in enumerate(np.unique(subject_names))}

    # return the corresponding group index
    return [unique_subject_names_dict[s] for s in subject_names]


def get_data_loader(X, y, batch_size=1):
    """
    Returns a data loader for some dataset.

    Args:
        X (np.array): Samples
        y (np.array): Labels
        batch_size (int): batch size used in this data loader
    Returns:
        (torch.DataLoader): complete data loader
    """
    # create pytorch dataset
    dataset = CoughDataset(torch.FloatTensor(X),
                           torch.FloatTensor(y))

    # get pytorch data loaders
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True)

    return data_loader
