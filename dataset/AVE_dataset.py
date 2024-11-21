import os
import h5py
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()

        self.split = split

        self.visual_feature_path = "./object_feature/" + self.split + "/feat"

        self.cnn14_feature_path = "./object_feature/cnn14_feat/"

        self.swin_feature_path = "./swinv2_large_feat/"

        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')

        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'right_labels.h5')

        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.raw_gt = pd.read_csv(data_root+"/Annotations.txt", sep="&", header=None)

        self.h5_isOpen = False


    def __getitem__(self, index):

        if not self.h5_isOpen:

            self.labels = h5py.File(self.labels_path, 'r')['avadataset']

            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True

        sample_index = self.sample_order[index]

        file_name = self.raw_gt.iloc[sample_index][1]

        visual_feat_path = self.visual_feature_path + '/' + file_name + '.npy'
        visual_feat = np.load(visual_feat_path)
        visual_feat = np.array(visual_feat, dtype=np.float64)

        audio_feat_path = self.cnn14_feature_path + file_name + '.npy'
        audio_feat = np.load(audio_feat_path)
        audio_feat = np.array(audio_feat, dtype=np.float64)

        label = self.labels[sample_index]

        return visual_feat, audio_feat, label

    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num


