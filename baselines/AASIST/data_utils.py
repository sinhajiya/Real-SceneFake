import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import torchaudio.functional as AF
from augmentations import audioaugment


def protocol_reader(protocol_path, is_eval=False):
    file_list = []
    labels = {}

    with open(protocol_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if is_eval:
                file_list.append(parts[0])
            else:
                path = " ".join(parts[:-1])
                label = int(parts[-1])
                file_list.append(path)
                labels[path] = label

    return file_list if is_eval else (labels, file_list)


def pad(x, max_len=64600):
    if x.shape[0] >= max_len:
        return x[:max_len]
    return np.tile(x, int(np.ceil(max_len / x.shape[0])))[:max_len]


def pad_random(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        start = np.random.randint(0, x_len - max_len + 1)
        return x[start:start + max_len]
    return np.tile(x, int(np.ceil(max_len / x_len)))[:max_len]


class BaseDataset(Dataset):
    def __init__(self, file_list, labels=None, cut=64600, sr=16000):
        self.file_list = file_list
        self.labels = labels
        self.cut = cut
        self.sr = sr
        self.hop = cut

        self.index_map = []
        self._prepare_index()

    def _prepare_index(self):
        for path in self.file_list:
            info = sf.info(path)
            total_len = int(info.frames * (self.sr / info.samplerate))

            for start in range(0, total_len, self.hop):
                self.index_map.append((path, start))

    def _load_audio(self, path):
        audio, sr = sf.read(path)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        audio = torch.tensor(audio, dtype=torch.float32)
        audio = AF.resample(audio, sr, self.sr)

        return audio.numpy()

    def _get_segment(self, audio, start):
        segment = audio[start:start + self.cut]
        if len(segment) < self.cut:
            segment = np.pad(segment, (0, self.cut - len(segment)))
        return segment.astype(np.float32)


class OurTrainDataset(BaseDataset):
    def __init__(self, file_list, labels, cut=64600, sr=16000, augmentations=None):
        super().__init__(file_list, labels, cut, sr)
        self.augmentations = augmentations or []

    def __len__(self):
        return len(self.index_map) * (2 if self.augmentations else 1)

    def __getitem__(self, index):
        use_aug = self.augmentations and index % 2 == 1
        base_index = index // 2 if self.augmentations else index

        path, start = self.index_map[base_index]

        audio = self._load_audio(path)  # LAZY LOAD HERE
        segment = self._get_segment(audio, start)

        if use_aug:
            segment = audioaugment(segment, self.sr, self.augmentations)

        segment = torch.from_numpy(segment)

        if segment.ndim != 1 or segment.numel() != self.cut:
            segment = torch.zeros(self.cut)

        return segment, self.labels[path]


class OurEvalDataset(BaseDataset):
    def __getitem__(self, index):
        path, start = self.index_map[index]

        audio = self._load_audio(path)  # LAZY LOAD HERE
        segment = torch.tensor(self._get_segment(audio, start))

        if self.labels is None:
            return segment, path

        return segment, self.labels[path], path
    
    def __len__(self):
        return len(self.index_map)
