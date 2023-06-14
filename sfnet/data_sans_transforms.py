import os
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import warnings
import torch.nn as nn
import warnings

from util import load_index, get_frames, qtile_normalize


class NeuralfpDataset(Dataset):
    def __init__(self, path, cfg, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.norm = cfg['norm']
        self.offset = cfg['offset']
        self.sample_rate = cfg['fs']
        self.dur = cfg['dur']
        self.n_frames = cfg
        self.size = cfg['train_size'] if train else cfg['val_size']
        self.filenames = load_index(path, max_len=self.size)

        self.ignore_idx = []
  
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = self.filenames[str(idx)]
        try:
            audio, sr = torchaudio.load(datapath)

        except Exception:

            print("Error loading:" + self.filenames[str(idx)])
            self.ignore_idx.append(idx)
            return self[idx+1]

        audio_mono = audio.mean(dim=0)
        if self.norm is not None:
            audio_mono = qtile_normalize(audio_mono, q=self.norm)
        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio_resampled = resampler(audio_mono)    

        clip_frames = int(self.sample_rate*self.dur)
        
        if len(audio_resampled) <= clip_frames:
            self.ignore_idx.append(idx)
            return self[idx + 1]
        
        #   For training pipeline, output a random frame of the audio
        if self.train:
            offset_mod = int(self.sample_rate*(self.offset) + clip_frames)
            if len(audio_resampled) < offset_mod:
                print(len(audio_resampled), offset_mod)
            r = np.random.randint(0,len(audio_resampled)-offset_mod)
            ri = np.random.randint(0,offset_mod - clip_frames)
            rj = np.random.randint(0,offset_mod - clip_frames)
            clip = audio_resampled[r:r+offset_mod]
            x_i = clip[ri:ri+clip_frames]
            x_j = clip[rj:rj+clip_frames]

            if self.transform is not None:
                x_i, x_j = self.transform(x_i, x_j)

            return torch.unsqueeze(x_i, 0), torch.unsqueeze(x_j, 0)
        
        #   For validation / test, output consecutive (overlapping) frames
        else:
            return torch.unsqueeze(audio_resampled, 0)
            # return audio_resampled
    
    def __len__(self):
        return len(self.filenames)