import os,glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip
import zipfile
import numpy as np
import io
import av
import math

#Custom modifications to the original code
from tqdm import tqdm
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler, BatchSampler
from torchvision.transforms import ToTensor
from decord import VideoReader
from utils import data_process
import json 
import pickle
import omegaconf
import lmdb

from utils import helper

"""
    All dataloader outputs should be a dict. 
    Follows a standard format. 
"""
data_mapper = {
    "readysteadygo": "/fs/cfar-projects/frequency_stuff/vid_inr/frames/readysteadygo_1080",
    "honeybee": "/fs/cfar-projects/frequency_stuff/vid_inr/frames/honeybee_1080",
    "beauty": "/fs/cfar-projects/frequency_stuff/vid_inr/frames/beauty_1080",
    "jockey": "/fs/cfar-projects/frequency_stuff/vid_inr/frames/jockey_1080",
    "shakendry": "/fs/cfar-projects/frequency_stuff/vid_inr/frames/shakendry_1080",
    "bosphore": "/fs/cfar-projects/frequency_stuff/vid_inr/frames/bosphore_1080",
    "yachtride": "/fs/cfar-projects/frequency_stuff/vid_inr/frames/yachtride_1080",
    "mario": "/fs/cfar-projects/frequency_stuff/long_videos/mario_variable/frames_all",
    "taichi_root": "/fs/cfar-projects/frequency_stuff/snap/nirvanapp/data/taichi_test_frames_interpolation/"
    }

class VideoFramesDataset(Dataset):

    def __init__(self,cfg,custom_transforms=None):
        
        self.cfg = cfg
        self.group_size = self.cfg.trainer.group_size #group_size
        self.batch_size = self.cfg.trainer.batch_size #batch_size
        self.stride = self.cfg.data.data_stride #stride #stride parameter, which specifies the distance between the start of consecutive groups.
        max_frames = cfg.data.max_frames

        #for interpolation experiments
        self.selected_frame_paths = None

        self.patch_shape = self.cfg.data.patch_shape
        if self.patch_shape is not None:
            self.patch_shape = omegaconf.OmegaConf.to_container(self.patch_shape) 
            self.patch_shape = self.patch_shape if isinstance(self.patch_shape, list) else (self.patch_shape, self.patch_shape)


        self.video_dir = self.cfg.data.data_path

        if not os.path.isdir(self.video_dir):
            self.video_dir = data_mapper[self.video_dir]

        self.transform = custom_transforms
        if self.transform is None:
            if self.cfg.data.data_shape is None:
                self.transform = transforms.ToTensor()
            else:
                resize_shape = omegaconf.OmegaConf.to_container(self.cfg.data.data_shape)
                self.transform = transforms.Compose([transforms.Resize(resize_shape),transforms.ToTensor()])
        

        self.frame_paths = sorted([os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir)])

        assert len(self.frame_paths) >= self.batch_size, 'Number of frames in video must be greater than or equal to batch size'


        if max_frames is not None:
            self.frame_paths = self.frame_paths[:max_frames] if max_frames is not None else self.frame_paths
            
        self.num_frames = len(self.frame_paths)        
        self.proc = data_process.DataProcessor(self.cfg.data,device='cpu') #dataloading on cpu

        assert self.group_size * self.batch_size <= self.num_frames, 'Number of frames in video must be greater than or equal to batch size * group size'

        #calculate length of dataset adter grouping
        self.num_groups = (self.num_frames - self.group_size)//(self.stride*self.group_size) + 1
        self.data_length = self.num_groups

        if self.selected_frame_paths is not None:
            self.data_length = len(self.selected_frame_paths)

    def __len__(self):
        return self.data_length
        
    def __getitem__(self, index):
        #index is the group id.

        #make it a dict. 
        batch = {}

        group_feats = []
        group_paths = []
        #load an entire group. 
        
        group_indices = list(range(0,len(self.frame_paths)))
        #divide into lists of size group_size
        group_indices = [group_indices[i:i + self.group_size] for i in range(0, len(group_indices), self.group_size)]
        for j in group_indices[index]:
            frame = Image.open(self.frame_paths[j]).convert('RGB')

            if self.transform is not None:
                frame = self.transform(frame)
                self.C,self.H,self.W = frame.shape
        
            feat,self.features_shape = self.proc.get_features(frame.unsqueeze(0),patch_shape=self.patch_shape)
            self.data_shape = frame.shape
            group_feats.append(feat.squeeze())
            group_paths.append(self.frame_paths[j])

        if self.cfg.data.data_format == 'patch_first':
            batch['features'] = torch.stack(group_feats, dim=0).permute(1,0,2,3,4)
        else:
            batch['features'] = torch.stack(group_feats, dim=0)

        batch['frame_ids'] = torch.tensor(group_indices[index])  #torch.tensor(list(range(index, index + self.group_size)))
        batch['group_id'] = torch.tensor(index)
        batch['features_shape'] = torch.tensor(self.features_shape) #also data shape but can differ due to padding while patching.
        batch['paths'] = group_paths
        batch['og_data_shape'] = torch.tensor(self.data_shape) # original data shape.

        return batch




def custom_collate_fn(batch):
    # Assuming each item in the batch is a dictionary
    # We initialize an empty batch dictionary
    collated_batch = {}
    # Get the first item's features_shape (since it's constant)
    collated_batch['features_shape'] = batch[0]['features_shape']
    collated_batch['og_data_shape'] = batch[0]['og_data_shape']

    if 'coords' in batch[0]:
        collated_batch['coords'] = batch[0]['coords']
        collated_batch['spatial_coord_idx'] = batch[0]['spatial_coord_idx']

    # Stack the data, frame_ids, and paths
    collated_batch['features'] = torch.stack([item['features'] for item in batch])
    collated_batch['frame_ids'] = torch.stack([item['frame_ids'] for item in batch])
    
    #collated_batch['group_id'] = torch.stack(torch.tensor([item['group_id'] for item in batch]))
    collated_batch['group_id'] = torch.tensor([item['group_id'] for item in batch])
    collated_batch['paths'] = [item['paths'] for item in batch]  # Assuming paths is non-tensor data

    if 'train_group_id' in batch[0]:
        collated_batch['train_group_id'] = torch.tensor([item['train_group_id'] for item in batch]) if 'train_group_id' in batch[0] else None

    return collated_batch

    

