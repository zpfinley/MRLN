import os
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pickle
import pdb
import librosa
import numpy as np
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip

class AVEDataset_VGG(Dataset):
    # for visual features extracted by VGG19 network
    def __init__(self, data_root, args=None, split='train'):
        super(AVEDataset_VGG, self).__init__()
        self.split = split

        # Now for the supervised task
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')

        # self.raw_audio_path = '/mnt/f/LAVISHData/AVE_Dataset/raw_audio/'
        self.raw_audio_path = '/mnt/f/LAVISHData/AVE_Dataset/new_audio_16k/'
        self.video_path = '/mnt/d/AVE_video/AVE_Dataset/AVE/'

        self.raw_gt = pd.read_csv(data_root + "/Annotations.txt", sep="&", header=None)

        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:

            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True

        sample_index = self.sample_order[index]

        file_name = self.raw_gt.iloc[sample_index][1]

        audio_wav_path = self.raw_audio_path + file_name + '.wav'

        video_path = self.video_path + file_name + '.mp4'

        return audio_wav_path, file_name, video_path

    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num

def get_audio_wav(video_path, save_pth, audio_name):
    '''
    :param name: 视频文件的路径
    :param save_pth: 保存目录
    :param audio_name: 音频名称
    :return:
    '''
    video = VideoFileClip(video_path)
    # print(video)
    audio = video.audio
    # print(save_pth)
    audio.write_audiofile(os.path.join(save_pth, audio_name+'.wav'), fps=16000)
    return "finish"

if __name__ == "__main__": 
    '''Dataset'''
    train_dataloader = torch.utils.data.DataLoader(
        AVEDataset_VGG('/mnt/d/AVE_data', None, split='val'),
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    for n_iter, batch_data in enumerate(train_dataloader):
        # if n_iter==10:
        #     break
        audio_wav_path, file_name, video_path = batch_data

        save_path = '/mnt/f/LAVISHData/AVE_Dataset/new_val_audio_16k'
        get_audio_wav(video_path[0], save_path, file_name[0])


