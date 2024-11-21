import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split

        self.visual_feature_path = "./object_feature/" + self.split + "/feat/"
        self.cnn14_feature_path = "./cnn14_feat/"

        self.swin_feature_path = "./swinv2_large_feat/"

        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'right_labels.h5') # original labels for testing

        self.bg_raw_gt = pd.read_csv(data_root + "/modified_noisydataset.txt", sep="&", header=None)

        self.raw_gt = pd.read_csv(data_root + "/Annotations.txt", sep="&", header=None)

        self.dir_labels_path = os.path.join(data_root, 'mil_labels.h5')  # video-level labels

        self.dir_labels_bg_path = os.path.join(data_root, 'labels_noisy.h5')  # only background

        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')

        self.h5_isOpen = False


    def __getitem__(self, index):

        if not self.h5_isOpen:

            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']


            self.clean_labels = h5py.File(self.dir_labels_path, 'r')['avadataset']

            if self.split == 'train':
                self.negative_labels = h5py.File(self.dir_labels_bg_path, 'r')['avadataset']
                self.negative_visual_feature = './object_feature/weak_feat/'
                self.negative_audio_feature = './cnn14_nosiy_feat_16k/'

            if self.split == 'test':
                self.labels = h5py.File(self.labels_path, 'r')['avadataset']

            self.h5_isOpen = True

        clean_length = len(self.sample_order)


        if index >= clean_length:

            valid_index = index - clean_length

            neg_file_name = self.bg_raw_gt.iloc[valid_index][1]

            neg_visual_feat_path = self.negative_visual_feature + neg_file_name + '.npy'
            # neg_visual_feat_path = self.swin_feature_path + neg_file_name + '.npy'

            visual_feat = np.load(neg_visual_feat_path)
            visual_feat = np.array(visual_feat, dtype=np.float64)

            neg_audio_feat_path = self.negative_audio_feature + neg_file_name + '.npy'
            audio_feat = np.load(neg_audio_feat_path)
            audio_feat = np.array(audio_feat, dtype=np.float64)

            label = self.negative_labels[valid_index]

        else:
            # test phase or negative training samples
            sample_index = self.sample_order[index]
            file_name = self.raw_gt.iloc[sample_index][1]
            visual_feat_path = self.visual_feature_path + file_name + '.npy'
            # visual_feat_path = self.swin_feature_path + file_name + '.npy'
            visual_feat = np.load(visual_feat_path)
            visual_feat = np.array(visual_feat, dtype=np.float64)

            audio_feat_path = self.cnn14_feature_path + file_name + '.npy'
            audio_feat = np.load(audio_feat_path)
            audio_feat = np.array(audio_feat, dtype=np.float64)


            if self.split == 'train':
                label = self.clean_labels[sample_index] # [29,]
            else:
                # for testing
                label = self.labels[sample_index]

        return visual_feat, audio_feat, label


    def __len__(self):
        if self.split == 'train':
            sample_order = h5py.File(self.sample_order_path, 'r')['order']
            noisy_labels = h5py.File(self.dir_labels_bg_path, 'r')['avadataset']
            # print(len(noisy_labels)) # 178
            length = len(sample_order) + len(noisy_labels)
        elif self.split == 'test':
            sample_order = h5py.File(self.sample_order_path, 'r')['order']
            length = len(sample_order)
        else:
            raise NotImplementedError

        return length


