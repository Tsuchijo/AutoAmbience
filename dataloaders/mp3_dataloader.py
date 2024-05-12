import os
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
import torch

class MP3Dataset(Dataset):
    def __init__(self, folder_path, sample_length, SR=44100):
        self.folder_path = folder_path
        self.sample_length = sample_length
        self.file_list = [file for file in os.listdir(folder_path)if '.mp3' in file]
        ## Load audio into memory
        self.data = [torch.from_numpy(librosa.load( self.folder_path + file , sr=SR, mono=False)[0] ) for file in self.file_list]
        self.data = torch.cat(self.data, dim=1)
        self.length = self.data.size(1) // self.sample_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[:, idx*self.sample_length:(idx+1)*self.sample_length]