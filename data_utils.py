import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import torchaudio.functional as AF
from augmentations import *
import random

def protocol_reader(protocol_path, is_eval=False):

    file_list = []
    labels = {}

    with open(protocol_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue  
            parts = line.split()
            if not is_eval:
                label = int(parts[-1])
                path = " ".join(parts[:-1])
            else:
                path = parts[0]

            file_list.append(path)
            if not is_eval:
                labels[path] = label

    if is_eval:
        return file_list
    else:
        return labels, file_list

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    # padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


# -- SceneFake loader --

class OurTrainDataset(Dataset):

    def __init__(self, file_list, labels, cut=64600, sr=16000, augmentations=[]):

        self.file_list = file_list
        self.labels = labels
        self.cut = cut
        self.sr = sr
        self.augmentations = augmentations
        # self.ADJUST_LOUDNESS = ADJUST_LOUDNESS
        self.hop = cut
        # self.ADD_SPEECH_PER = ADD_SPEECH_PER
        self.audio_cache = {}
        self.index_map = []

        self._prepare_audio()
        # augs = []
        # if self.ADD_NOISE_FLOOR: augs.append("Noise")
        # if self.ADJUST_LOUDNESS: augs.append("Loudness")
        # if self.ADD_SPEECH_PER: augs.append("SPEECH")

        print("Augmentations:", augmentations if augmentations else "None")


    def _prepare_audio(self):

        for path in self.file_list:

            audio, sr = sf.read(path)
            # print("before:", sr, "samples:", len(audio))
            if len(audio.shape) == 2:
                audio = np.mean(audio, axis=1)

            audio = torch.tensor(audio, dtype=torch.float32)

            # if sr != self.sr:
            audio = AF.resample(audio, sr, self.sr)
            # print("samples after resample:", audio.shape[0])
            # print("duration seconds:", audio.shape[0] / self.sr)
            audio = audio.numpy()

            self.audio_cache[path] = audio

            total_len = len(audio)

            start = 0

            while start < total_len:
                self.index_map.append((path, start))
                start += self.hop

    # def __len__(self):
    #     return len(self.index_map)
    
    def __len__(self):
        if self.augmentations:
            return 2 * len(self.index_map)
        return len(self.index_map)
    
    
    def __getitem__(self, index):

        if self.augmentations:
            base_index = index // 2
            use_aug = index % 2 == 1
        else:
            base_index = index
            use_aug = False

        path, start = self.index_map[base_index]
        audio = self.audio_cache[path]

        segment = audio[start:start + self.cut]

        if len(segment) < self.cut:
            segment = np.pad(segment, (0, self.cut - len(segment)))

        if use_aug:
            segment = audioaugment(segment, self.sr, self.augmentations)

        segment = torch.from_numpy(segment).float()

        if segment.dim() != 1 or segment.numel() != self.cut:
            segment = torch.zeros(self.cut)

        label = self.labels[path]

        return segment, label


    # def __getitem__(self, index):

    #     path, start = self.index_map[index]
    #     audio = self.audio_cache[path]

    #     segment = audio[start:start + self.cut]

    #     if len(segment) < self.cut:
    #         segment = np.pad(segment, (0, self.cut - len(segment)))

    #     segment = torch.from_numpy(segment).float()

    #     label = self.labels[path]

        

    #     # if self.ADD_NOISE_FLOOR and random.random() < 0.5:
    #     #     segment = add_noise_floor(segment)

    #     # if self.ADJUST_LOUDNESS and random.random() < 0.5:
    #     #     segment = loudness_adjust_file(segment)

    #     # if self.ADD_SPEECH_PER and random.random() < 0.5:
    #     #     segment = add_speech_per(segment, self.sr, target_len=self.cut)

    #     if segment.dim() != 1 or segment.numel() != self.cut:
    #         segment = torch.zeros(self.cut)

    #     return segment, label


class OurEvalDataset(Dataset):

    def __init__(self, file_list, labels=None, cut=64600, sr=16000):

        self.file_list = file_list
        self.labels = labels
        self.cut = cut
        self.sr = sr
        self.hop = cut

        self.audio_cache = {}
        self.index_map = []

        self._prepare_audio()

    def _prepare_audio(self):

        for path in self.file_list:

            audio, sr = sf.read(path)

            if len(audio.shape) == 2:
                audio = np.mean(audio, axis=1)

            audio = torch.tensor(audio, dtype=torch.float32)
# resample to 16k taaki sb consistent rhe 
            # if sr != self.sr:
            audio = AF.resample(audio, sr, self.sr)

            audio = audio.numpy()

            self.audio_cache[path] = audio

            total_len = len(audio)
# cut 
            start = 0
            while start < total_len:
                self.index_map.append((path, start))
                start += self.hop

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):

        path, start = self.index_map[index]

        audio = self.audio_cache[path]

        segment = audio[start:start + self.cut]

        if len(segment) < self.cut:
            segment = np.pad(segment, (0, self.cut - len(segment)))

        segment = segment.astype(np.float32)

        audio_tensor = torch.tensor(segment)

        if self.labels is None:
            return audio_tensor, path

        label = self.labels[path]

        return audio_tensor, label, path