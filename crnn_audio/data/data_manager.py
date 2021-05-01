import os

import torch
import torch.utils.data as tdata
from sklearn.model_selection import train_test_split

from src.utils.get_data import import_data
from crnn_audio.data.data_sets import FolderDataset
from crnn_audio.utils.util import load_audio
import pandas as pd


class CSVDataManager(object):

    def __init__(self, config):
        load_formats = {
            'audio': load_audio
        }

        assert config['format'] in load_formats, "Pass valid data format"

        self.dir_path = config['path']
        self.loader_params = config['loader']

        self.load_func = load_formats[config['format']]

        M_PATH = '../data'
        _, self.metadata_df = import_data(M_PATH, drop_user_features=True, segmentation_type='no', return_type='pd')
        self.metadata_df['cough_type'] = self.metadata_df.apply(lambda row: 'wet' if row['Label'] == 1 else 'dry',
                                                                axis=1)
        self.classes = self._get_classes(
            self.metadata_df[['cough_type', 'Label']])
        self.data_splits = self._split_data(self.metadata_df)

    @staticmethod
    def _get_classes(df):
        c_col = df.columns[0]
        idx_col = df.columns[1]
        return df.drop_duplicates().sort_values(idx_col)[c_col].unique()

    def _split_data(self, df):
        ret = {"train": [], "val": []}
        df_tr, df_val = train_test_split(df, test_size=0.3, random_state=42)

        # oversampling to deal with class imbalance
        lst = [df_tr]
        max_size = df_tr['cough_type'].value_counts().max()
        for class_index, group in df_tr.groupby('cough_type'):
            lst.append(group.sample(max_size - len(group), replace=True))
        df_tr = pd.concat(lst)
        df_tr = df_tr.sample(frac=1)

        for row in df_tr[['Label', 'cough_type']].iterrows():
            f_name = os.path.join(self.dir_path, 'wav_data', f'{row[0]}.wav')
            ret["train"].append(
                {'path': f_name, 'class': row[1]['cough_type'], 'class_idx': row[1]['Label']})
        for row in df_val[['Label', 'cough_type']].iterrows():
            f_name = os.path.join(self.dir_path, 'wav_data', f'{row[0]}.wav')
            ret["val"].append(
                {'path': f_name, 'class': row[1]['cough_type'], 'class_idx': row[1]['Label']})
        return ret

    def get_loader(self, name, transfs):
        assert name in self.data_splits
        dataset = FolderDataset(
            self.data_splits[name], load_func=self.load_func, transforms=transfs)

        return tdata.DataLoader(dataset=dataset, **self.loader_params, collate_fn=self.pad_seq)

    @staticmethod
    def pad_seq(batch):
        # sort_ind should point to length
        sort_ind = 0
        sorted_batch = sorted(
            batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels = zip(*sorted_batch)

        lengths, srs, labels = map(
            torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels])

        # seqs_pad -> (batch, time, channel)
        seqs_pad = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)
        # seqs_pad = seqs_pad_t.transpose(0, 1)
        return seqs_pad, lengths, srs, labels


if __name__ == '__main__':
    pass
