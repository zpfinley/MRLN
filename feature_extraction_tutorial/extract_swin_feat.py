import os
import timm
import h5py
import torch
import pandas as pd
import numpy as np
import glob
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.raw_gt = pd.read_csv(data_root+"/Annotations.txt", sep="&", header=None)
        self.video_folder = "/AVE_Dataset/video_frames"
        self.my_normalize = Compose([Resize([192, 192],
                                    interpolation=Image.BICUBIC),
                                     ToTensor(),
                                     Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        self.h5_isOpen = False


    def __getitem__(self, index):
        if not self.h5_isOpen:

            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True


        sample_index = self.sample_order[index]
        file_name = self.raw_gt.iloc[sample_index][1]

        total_img = []
        for vis_idx in range(10):
            img_path = self.video_folder + '/' + file_name + '/' + str("{:04d}".format(vis_idx)) + '.jpg'
            tmp_img = Image.open(img_path).convert('RGB')
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)

        return total_img, file_name


    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num

model = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
model = model.cuda()
model = model.eval()

if __name__ == '__main__':

    train_dataloader = DataLoader(
            AVEDataset('/AVE_data/', split='train'),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )

    for n_iter, batch_data in enumerate(train_dataloader):

        visual_feat, video_name = batch_data

        visual_feat = visual_feat.cuda()
        bz, t, c, h, w = visual_feat.shape
        visual_feat = visual_feat.view(bz*t, c, h, w)

        with torch.no_grad():
            save_dir = "/swinv2_large_feat"
            outfile = os.path.join(save_dir, video_name[0] + '.npy')
            if os.path.exists(outfile):
                print("文件已经存在")
                # break
            else:
                image_features = model.forward_features(visual_feat)
                image_features = image_features.cpu().numpy()
                print(outfile)
                np.save(outfile, image_features)






