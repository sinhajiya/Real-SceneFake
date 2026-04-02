# import numpy as np
# from torch import Tensor, Generator
# import librosa
# from torch.utils.data import Dataset, DataLoader
# # from collection impoer 
# from collections import defaultdict
# from RawBoost import  process_Rawboost_feature	
# from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
# import random
# from pathlib import Path
# import os
# import soundfile as sf
# import torch
# import torchaudio.functional as AF
# from utils import *
# # original file from 

# def protocol_reader(protocol_path, is_eval=False):

#     file_list = []
#     labels = {}

#     with open(protocol_path, "r") as f:
#         for line in f:
#             line = line.strip()

#             if not line:
#                 continue  
#             parts = line.split()
#             if not is_eval:
#                 label = int(parts[-1])
#                 path = " ".join(parts[:-1])
#             else:
#                 path = parts[0]

#             file_list.append(path)
#             if not is_eval:
#                 labels[path] = label

#     if is_eval:
#         return file_list
#     else:
#         return labels, file_list


# def pad(x, max_len=64600):
#     x_len = x.shape[0]
#     if x_len >= max_len:
#         return x[:max_len]
#     # need to pad
#     num_repeats = int(max_len / x_len)+1
#     padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
#     return padded_x	

# # -- SceneFake loader --

# class OurTrainDataset(Dataset):
# # take 4secs audio non overlapping and augmentations 
# # return segment and label
#     def __init__(self, file_list, labels, cut=64600, sr=16000, augmentations=[]):

#         self.file_list = file_list
#         self.labels = labels
#         self.cut = cut
#         self.sr = sr
#         self.augmentations = augmentations
#         self.hop = cut
#         self.audio_cache = {}
#         self.index_map = []

#         self._prepare_audio()
        
#         print("Augmentations:", augmentations if augmentations else "None")


#     def _prepare_audio(self):

#         for path in self.file_list:

#             audio, sr = sf.read(path)
#             # print("before:", sr, "samples:", len(audio))
#             if len(audio.shape) == 2:
#                 audio = np.mean(audio, axis=1)

#             audio = torch.tensor(audio, dtype=torch.float32)

#             # if sr != self.sr:
#             audio = AF.resample(audio, sr, self.sr)
#             # print("samples after resample:", audio.shape[0])
#             # print("duration seconds:", audio.shape[0] / self.sr)
#             audio = audio.numpy()

#             self.audio_cache[path] = audio

#             total_len = len(audio)

#             start = 0

#             while start < total_len:
#                 self.index_map.append((path, start))
#                 start += self.hop


#     def __len__(self):
#         if self.augmentations:
#             return 2 * len(self.index_map)
#         return len(self.index_map)
    
    
#     def __getitem__(self, index):

#         if self.augmentations:
#             base_index = index // 2
#             use_aug = index % 2 == 1
#         else:
#             base_index = index
#             use_aug = False

#         path, start = self.index_map[base_index]
#         audio = self.audio_cache[path]

#         segment = audio[start:start + self.cut]

#         if len(segment) < self.cut:
#             segment = np.pad(segment, (0, self.cut - len(segment)))

#         if use_aug:
#             segment = audioaugment(segment, self.sr, self.augmentations)

#         segment = torch.from_numpy(segment).float()

#         if segment.dim() != 1 or segment.numel() != self.cut:
#             segment = torch.zeros(self.cut)

#         label = self.labels[path]

#         return segment, label


# class OurEvalDataset(Dataset):
# # returns segment, label and path taaki per utterance eval ho paye. 
#     def __init__(self, file_list, labels=None, cut=64600, sr=16000):

#         self.file_list = file_list
#         self.labels = labels
#         self.cut = cut
#         self.sr = sr
#         self.hop = cut

#         self.audio_cache = {}
#         self.index_map = []

#         self._prepare_audio()

#     def _prepare_audio(self):

#         for path in self.file_list:

#             audio, sr = sf.read(path)

#             if len(audio.shape) == 2:
#                 audio = np.mean(audio, axis=1)

#             audio = torch.tensor(audio, dtype=torch.float32)
# # resample to 16k taaki sb consistent rhe 
#             # if sr != self.sr:
#             audio = AF.resample(audio, sr, self.sr)

#             audio = audio.numpy()

#             self.audio_cache[path] = audio

#             total_len = len(audio)
# # cut 
#             start = 0
#             while start < total_len:
#                 self.index_map.append((path, start))
#                 start += self.hop

#     def __len__(self):
#         return len(self.index_map)

#     def __getitem__(self, index):

#         path, start = self.index_map[index]

#         audio = self.audio_cache[path]

#         segment = audio[start:start + self.cut]

#         if len(segment) < self.cut:
#             segment = np.pad(segment, (0, self.cut - len(segment)))

#         segment = segment.astype(np.float32)

#         audio_tensor = torch.tensor(segment)

#         if self.labels is None:
#             return audio_tensor, path

#         label = self.labels[path]

#         return audio_tensor, label, path


# class Dataset_train_FT(Dataset):
#     def __init__(self, protocol_path, k, cut=64600, sr=16000):

#         self.cut = cut
#         self.sr = sr
#         self.hop = cut
#         self.audio_cache = {}
#         self.index_map = []

#         protocol_file = Path(protocol_path) / "train.txt"
#         print("Reading:", protocol_file)

#         lines = open(protocol_file).readlines()

#         class_files = defaultdict(list)

#         for line in lines:
#             file_path, label = line.strip().split()
#             label = int(label)
#             class_files[label].append(file_path)
#             # print(f"Found file for label {label}: {file_path}")

#         selected_files = []
#         self.labels = {}

#         for label in class_files:
#             files = class_files[label]
#             print(len(files))
#             if len(files) >= k:
#                 chosen = random.sample(files, k)
#             else:
#                 chosen = random.choices(files, k=k)

#             for f in chosen:
#                 selected_files.append(f)
#                 self.labels[f] = label
#                 print(f"Selected for label {label}: {f}")

#         self.file_list = selected_files

#         print(f"K-shot → {k} real + {k} fake")
#         print("Total selected files:", len(self.file_list))

#         self._prepare_audio()

#     def _prepare_audio(self):

#         for path in self.file_list:

#             audio, sr = sf.read(path)

#             if len(audio.shape) == 2:
#                 audio = np.mean(audio, axis=1)

#             audio = torch.tensor(audio, dtype=torch.float32)

#             audio = AF.resample(audio, sr, self.sr)
#             audio = audio.numpy()

#             self.audio_cache[path] = audio

#             total_len = len(audio)
#             start = 0

#             while start < total_len:
#                 self.index_map.append((path, start))
#                 start += self.hop

#     def __len__(self):
#         return len(self.index_map)

#     def __getitem__(self, index):

#         path, start = self.index_map[index]
#         audio = self.audio_cache[path]

#         segment = audio[start:start + self.cut]

#         if len(segment) < self.cut:
#             segment = np.pad(segment, (0, self.cut - len(segment)))

#         segment = torch.from_numpy(segment).float()

#         if segment.dim() != 1 or segment.numel() != self.cut:
#             segment = torch.zeros(self.cut)

#         label = self.labels[path]

#         return segment, label


# def get_loader(seed: int, args):

#     gen = Generator()
#     gen.manual_seed(seed)
#     pp = Path(args.protocol_path)
#     train_protocol = pp / "train.txt"
#     val_protocol = pp / "val.txt"


#     if not args.eval:
#         train_labels, train_files = protocol_reader(train_protocol)
#         val_labels, val_files = protocol_reader(val_protocol)

#         print("train files:", len(train_files))
#         print("validation files:", len(val_files))
#         print(type(train_labels))
#         labels_array = np.array(list(train_labels.values()))
#         class_counts = np.bincount(labels_array)
#         print("Training class counts:", class_counts)
#         total = len(labels_array)
#         class_weights = total / (len(class_counts) * class_counts)

#         print("Class weights:", class_weights)

#         if getattr(args, "k_value_finetune", False):

#             train_set = Dataset_train_FT(args.protocol_path, args.k_value_finetune)
#         else:
#             train_set = OurTrainDataset(
#                 file_list=train_files,
#                 labels=train_labels,
#                 # augmentations =args.augmentations
            
#             )

#         trn_loader = DataLoader(
#             train_set,
#             batch_size=args.batch_size,
#             shuffle=True,
#             drop_last=True,
#             pin_memory=True,
#             num_workers=args.num_workers,
#             worker_init_fn=seed_worker,
#             generator=gen
#         )

#         val_set = OurEvalDataset(
#             file_list=val_files,
#             labels=val_labels
#         )

#         val_loader = DataLoader(
#             val_set,
#             batch_size=args.batch_size,
#             shuffle=False,
#             drop_last=False,
#             pin_memory=True,
#             num_workers=args.num_workers

#         )
#     else:
#         # trn_loader, val_loader = None, None
#         print("evalllll loaderrrrrr")

#         if not args.scenefake_eval:
#             print("not sf eval worked!")
#             unseen_protocol = pp / "unseen_test.txt"
#             seen_protocol = pp/ "seen_test.txt"

#             seen_labels, seen_files = protocol_reader(seen_protocol)
#             print("seen test files:", len(seen_files))

#             seen_set = OurEvalDataset(
#                 file_list=seen_files,
#                 labels=seen_labels
#             )
#             seen_loader = DataLoader(
#                 seen_set,
#                 batch_size=args.batch_size,
#                 shuffle=False,
#                 drop_last=False,
#                 pin_memory=True,
#                 num_workers=args.num_workers
#             )

#             if os.path.exists(unseen_protocol):
#                 unseen_labels, unseen_files = protocol_reader(unseen_protocol)

#                 print("unseen test files:", len(unseen_files))

#                 unseen_set = OurEvalDataset(
#                     file_list=unseen_files,
#                     labels=unseen_labels
#                 )

#                 unseen_loader = DataLoader(
#                     unseen_set,
#                     batch_size=args.batch_size,
#                     shuffle=False,
#                     drop_last=False,
#                     pin_memory=True,
#                     num_workers=args.num_workers
#                 )
#             else:
#                 unseen_loader = None

#             return None, None, seen_loader, unseen_loader, None
        
#         else:
#             print("main  sf eval and protocols wale tk pahcuh gya huuuu")
#             test_protocols = args.sf_protocols
#             seen_loader = []
#             for num, t in enumerate(test_protocols):
#                 seen_labels, seen_files = protocol_reader(t)
#                 print("test files in fold :", num+1, " is ", len(seen_files))
#                 seen_set = OurEvalDataset(
#                 file_list=seen_files,
#                 labels=seen_labels
#             )
#                 seen_loader.append(DataLoader(
#                 seen_set,
#                 batch_size=args.batch_size,
#                 shuffle=False,
#                 drop_last=False,
#                 pin_memory=True,
#                 num_workers=args.num_workers
#             )      )
                
#             return None, None, seen_loader, None, None



#     return trn_loader, val_loader, None, None, class_weights



import numpy as np
from torch import Generator
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from pathlib import Path
import os
import soundfile as sf
import torch
import torchaudio.functional as AF
from utils import *
from RawBoost import process_Rawboost_feature
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav


# -----------------------------
# Protocol Reader
# -----------------------------
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
                labels[path] = label
            else:
                path = parts[0]

            file_list.append(path)

    return file_list if is_eval else (labels, file_list)


# -----------------------------
# Base Audio Loader
# -----------------------------
def load_audio(path, target_sr):
    audio, sr = sf.read(path)

    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)

    audio = torch.from_numpy(audio).float()
    if sr != target_sr:
        audio = AF.resample(audio, sr, target_sr)

    return audio


def get_segments(audio_len, cut):
    num_chunks = (audio_len + cut - 1) // cut
    return [i * cut for i in range(num_chunks)]


# -----------------------------
# Train Dataset
# -----------------------------
class OurTrainDataset(Dataset):
    def __init__(self, file_list, labels, cut=64600, sr=16000, augmentations=[]):
        self.file_list = file_list
        self.labels = labels
        self.cut = cut
        self.sr = sr
        self.augmentations = augmentations

        self.index_map = []
        self._prepare_index()

        print("Augmentations:", augmentations if augmentations else "None")

    def _prepare_index(self):
        for path in self.file_list:
            info = sf.info(path)
            total_len = int(info.frames * self.sr / info.samplerate)
            for start in get_segments(total_len, self.cut):
                self.index_map.append((path, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        path, start = self.index_map[index]

        audio = load_audio(path, self.sr)

        segment = audio[start:start + self.cut]

        if segment.shape[0] < self.cut:
            padded = torch.zeros(self.cut)
            padded[:segment.shape[0]] = segment
            segment = padded

        # probabilistic augmentation (kept same logic capability)
        if self.augmentations and random.random() < 0.5:
            segment_np = segment.numpy()
            segment_np = audioaugment(segment_np, self.sr, self.augmentations)
            segment = torch.from_numpy(segment_np).float()

        if segment.dim() != 1 or segment.numel() != self.cut:
            segment = torch.zeros(self.cut)

        label = self.labels[path]
        return segment, label


# -----------------------------
# Eval Dataset
# -----------------------------
class OurEvalDataset(Dataset):
    def __init__(self, file_list, labels=None, cut=64600, sr=16000):
        self.file_list = file_list
        self.labels = labels
        self.cut = cut
        self.sr = sr

        self.index_map = []
        self._prepare_index()

    def _prepare_index(self):
        for path in self.file_list:
            info = sf.info(path)
            total_len = int(info.frames * self.sr / info.samplerate)
            for start in get_segments(total_len, self.cut):
                self.index_map.append((path, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        path, start = self.index_map[index]

        audio = load_audio(path, self.sr)
        segment = audio[start:start + self.cut]

        if segment.shape[0] < self.cut:
            padded = torch.zeros(self.cut)
            padded[:segment.shape[0]] = segment
            segment = padded

        if self.labels is None:
            return segment, path

        label = self.labels[path]
        return segment, label, path


# -----------------------------
# K-shot Dataset
# -----------------------------
class Dataset_train_FT(Dataset):
    def __init__(self, protocol_path, k, cut=64600, sr=16000):
        self.cut = cut
        self.sr = sr

        protocol_file = Path(protocol_path) / "train.txt"
        lines = open(protocol_file).readlines()

        class_files = defaultdict(list)

        for line in lines:
            file_path, label = line.strip().split()
            class_files[int(label)].append(file_path)

        self.file_list = []
        self.labels = {}

        for label, files in class_files.items():
            chosen = random.sample(files, k) if len(files) >= k else random.choices(files, k=k)
            for f in chosen:
                self.file_list.append(f)
                self.labels[f] = label

        self.index_map = []
        self._prepare_index()

    def _prepare_index(self):
        for path in self.file_list:
            info = sf.info(path)
            total_len = int(info.frames * self.sr / info.samplerate)
            for start in get_segments(total_len, self.cut):
                self.index_map.append((path, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        path, start = self.index_map[index]

        audio = load_audio(path, self.sr)
        segment = audio[start:start + self.cut]

        if segment.shape[0] < self.cut:
            padded = torch.zeros(self.cut)
            padded[:segment.shape[0]] = segment
            segment = padded

        if segment.dim() != 1 or segment.numel() != self.cut:
            segment = torch.zeros(self.cut)

        label = self.labels[path]
        return segment, label


# -----------------------------
# Loader
# -----------------------------
def get_loader(seed: int, args):
    gen = Generator()
    gen.manual_seed(seed)

    pp = Path(args.protocol_path)
    train_protocol = pp / "train.txt"
    val_protocol = pp / "val.txt"

    num_workers = args.num_workers if args.num_workers else max(1, os.cpu_count() // 2)

    if not args.eval:
        train_labels, train_files = protocol_reader(train_protocol)
        val_labels, val_files = protocol_reader(val_protocol)

        labels_array = np.array(list(train_labels.values()))
        class_counts = np.bincount(labels_array)
        total = len(labels_array)
        class_weights = total / (len(class_counts) * class_counts)

        if getattr(args, "k_value_finetune", False):
            train_set = Dataset_train_FT(args.protocol_path, args.k_value_finetune)
        else:
            train_set = OurTrainDataset(
                file_list=train_files,
                labels=train_labels
            )

        trn_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=gen
        )

        val_set = OurEvalDataset(
            file_list=val_files,
            labels=val_labels
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2
        )

        return trn_loader, val_loader, None, None, class_weights

    else:
        if not args.scenefake_eval:
            unseen_protocol = pp / "unseen_test.txt"
            seen_protocol = pp / "seen_test.txt"

            seen_labels, seen_files = protocol_reader(seen_protocol)

            seen_set = OurEvalDataset(seen_files, seen_labels)
            seen_loader = DataLoader(
                seen_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True
            )

            if os.path.exists(unseen_protocol):
                unseen_labels, unseen_files = protocol_reader(unseen_protocol)
                unseen_set = OurEvalDataset(unseen_files, unseen_labels)

                unseen_loader = DataLoader(
                    unseen_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=False,
                    persistent_workers=True
                )
            else:
                unseen_loader = None

            return None, None, seen_loader, unseen_loader, None

        else:
            seen_loader = []
            for t in args.sf_protocols:
                seen_labels, seen_files = protocol_reader(t)

                seen_set = OurEvalDataset(seen_files, seen_labels)
                seen_loader.append(
                    DataLoader(
                        seen_set,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=False,
                        persistent_workers=True
                    )
                )

            return None, None, seen_loader, None, None