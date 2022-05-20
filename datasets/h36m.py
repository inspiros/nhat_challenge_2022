import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['H36MRestorationDataset']


class H36MRestorationDataset(Dataset):
    train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    test_subjects = ['S9', 'S11']
    actions = ['Directions', 'Directions1', 'Discussion', 'Discussion1',
               'Eating', 'Eating2', 'Greeting', 'Greeting1', 'Phoning',
               'Phoning1', 'Photo', 'Photo1', 'Posing', 'Posing1',
               'Purchases', 'Purchases1', 'Sitting1', 'Sitting2',
               'SittingDown', 'SittingDown2', 'Smoking', 'Smoking1',
               'Waiting', 'Waiting1', 'WalkDog', 'WalkDog1', 'WalkTogether',
               'WalkTogether1', 'Walking', 'Walking1']
    n_cams = 4
    resolution = (1000, 1000)

    def __init__(self,
                 source_file_path,
                 target_file_path=None,
                 sample_length=1,
                 sample_step=1,
                 partition=None,
                 transform=None,
                 target_transform=None,
                 return_sample_meta=False):
        self.source_file_path = source_file_path
        self.target_file_path = target_file_path
        self.sample_length = int(sample_length)
        self.sample_step = int(sample_step)
        self.partition = partition
        self.transform = transform
        self.target_transform = target_transform
        self.return_sample_meta = return_sample_meta

        assert self.partition in [None, 'train', 'test']

        source_data = np.load(self.source_file_path, allow_pickle=True)['positions_2d'].item()
        self.source_data = self._preprocess(source_data)
        if self.target_file_path is not None:
            target_data = np.load(self.target_file_path, allow_pickle=True)['positions_2d'].item()
            self.target_data = self._preprocess(target_data)
        else:
            self.target_data = None

        self.subjects = sorted(list(self.source_data.keys()))

        self.sample_metas = []

        for subject in self.subjects:
            for action in self.actions:
                for cam in range(self.n_cams):
                    if action not in self.source_data[subject] or (
                            self.target_data is not None and action not in self.target_data[subject]
                    ):
                        continue
                    sequence_length = min(
                        len(self.source_data[subject][action][cam]),
                        len(self.target_data[subject][action][cam])
                        if self.target_data is not None else np.inf,
                    )
                    for frame_id in range(0, sequence_length - self.sample_length, self.sample_step):
                        self.sample_metas.append((subject, action, cam, frame_id))
                    # if frame_id != sequence_length - 1 - self.sample_length:
                    #     self.sample_metas.append((subject, action, cam, sequence_length - 1 - self.sample_length))

    def _preprocess(self, data):
        # remove out-of-partition subjects
        if self.partition is not None:
            if self.partition == 'train':
                keep_subjects = self.train_subjects
            elif self.partition == 'test':
                keep_subjects = self.test_subjects
            for subject in set(data.keys()).difference(keep_subjects):
                data.pop(subject)
        # remove space in action names
        for subject_data in data.values():
            need_renamed_actions = []
            for k, v in subject_data.items():
                if ' ' in k:
                    need_renamed_actions.append(k)
            for k in need_renamed_actions:
                subject_data[k.replace(' ', '')] = subject_data.pop(k)
        return data

    def _action_to_idx(self, action):
        return self.actions.index(action)

    def __len__(self):
        return len(self.sample_metas)

    def __getitem__(self, item):
        subject, action, cam, frame_id = self.sample_metas[item]
        source_data = self.source_data[subject][action][cam][frame_id:frame_id + self.sample_length]
        source_data = torch.tensor(source_data, dtype=torch.float32)
        if self.target_data is not None:
            target_data = self.target_data[subject][action][cam][frame_id:frame_id + self.sample_length]
            target_data = torch.tensor(target_data, dtype=torch.float32)
        else:
            target_data = source_data.clone()

        # transforms
        if self.transform is not None:
            source_data = self.transform(source_data)
        if self.target_transform is not None:
            target_data = self.target_transform(target_data)

        if self.return_sample_meta:
            return source_data, target_data, subject, self._action_to_idx(action), cam, frame_id
        return source_data, target_data
