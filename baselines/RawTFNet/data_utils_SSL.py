import numpy as np
from torch import Tensor, Generator
import librosa
from torch.utils.data import Dataset, DataLoader

from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import random
from pathlib import Path
import os
import soundfile as sf
import torch
import torchaudio.functional as AF
from utils import *
# original file from 
import torchaudio.transforms as T

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
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

# -- SceneFake loader --

class OurTrainDataset(Dataset):
# take 4secs audio non overlapping and augmentations 
# return segment and label
    def __init__(self, file_list, labels, cut=64600, sr=16000, augmentations=[]):

        self.file_list = file_list
        self.labels = labels
        self.cut = cut
        self.sr = sr
        self.augmentations = augmentations
        self.hop = cut
        self.audio_cache = {}
        self.index_map = []

        self._prepare_audio()
        
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


class OurEvalDataset(Dataset):
# returns segment, label and path taaki per utterance eval ho paye. 
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
            # audio = AF.resample(audio, sr, self.sr)
            resampler = T.Resample(orig_freq=sr, new_freq=self.sr)
            audio = resampler(audio)

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

#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
    elif algo==9: # parallel (1+2+3)
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)
        feature3=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
        feature_para=feature1+feature2+feature3
        feature=normWav(feature_para,0)  #normalized resultant waveform
    elif algo==10:
        LNL_probability = random.random()
        ISD_probability = random.random()
        if LNL_probability > args.LnL_ratio:
            if ISD_probability > args.ISD_ratio: # no aug
                pass
            else: # ISD
                feature = ISD_additive_noise(feature, args.P, args.g_sd)
        else:
            if ISD_probability > args.ISD_ratio: # LNL
                feature = LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
            else: # both
                feature1 = LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                         args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
                feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
                feature_para = feature1 + feature2
                feature = normWav(feature_para,0)    # original data without Rawboost processing           
    elif algo == 11:
        LNL_probability = random.random()
        ISD_probability = random.random()
        SSI_probability = random.random()
        if LNL_probability < args.LnL_ratio:
            feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        if ISD_probability < args.ISD_ratio:
            feature = ISD_additive_noise(feature, args.P, args.g_sd)
        if SSI_probability < args.SSI_ratio:
            feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                         args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG,
                                         args.maxG, sr)
    else:
        
        feature=feature
    
    return feature





def get_loader(seed: int, args):

    gen = Generator()
    gen.manual_seed(seed)
    pp = Path(args.protocols_path)
    train_protocol = pp / "train.txt"
    val_protocol = pp / "val.txt"


    if not args.eval:
        train_labels, train_files = protocol_reader(train_protocol)
        val_labels, val_files = protocol_reader(val_protocol)

        print("train files:", len(train_files))
        print("validation files:", len(val_files))
        print(type(train_labels))
        labels_array = np.array(list(train_labels.values()))
        class_counts = np.bincount(labels_array)
        print("Training class counts:", class_counts)
        total = len(labels_array)
        class_weights = total / (len(class_counts) * class_counts)

        print("Class weights:", class_weights)

        # print(train_labels.head())
        # print(dict(itertools.islice(train_labels.items(), 2)))

        # class_counts = np.bincount(train_labels)
        # print("Training class counts:", class_counts)

        # total = len(train_labels)
        # class_weights = total / (len(class_counts) * class_counts)

        # print("Class weights:", class_weights)

        train_set = OurTrainDataset(
            file_list=train_files,
            labels=train_labels,
            # augmentations =args.augmentations
        
        )

        trn_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers,
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
            pin_memory=True,
            num_workers=args.num_workers

        )
    else:
        # trn_loader, val_loader = None, None
        print("evalllll loaderrrrrr")

        if not args.scenefake_eval:
            print("not sf eval worked!")
            unseen_protocol = pp / "unseen_test.txt"
            seen_protocol = pp/ "seen_test.txt"

            seen_labels, seen_files = protocol_reader(seen_protocol)
            print("seen test files:", len(seen_files))

            seen_set = OurEvalDataset(
                file_list=seen_files,
                labels=seen_labels
            )
            seen_loader = DataLoader(
                seen_set,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=args.num_workers
            )

            if os.path.exists(unseen_protocol):
                unseen_labels, unseen_files = protocol_reader(unseen_protocol)

                print("unseen test files:", len(unseen_files))

                unseen_set = OurEvalDataset(
                    file_list=unseen_files,
                    labels=unseen_labels
                )

                unseen_loader = DataLoader(
                    unseen_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=args.num_workers
                )
            else:
                unseen_loader = None

            return None, None, seen_loader, unseen_loader, None
        
        else:
            print("main  sf eval and protocols wale tk pahcuh gya huuuu")
            test_protocols = args.sf_eval_protocols
            seen_loader = []
            for num, t in enumerate(test_protocols):
                seen_labels, seen_files = protocol_reader(t)
                print("test files in fold :", num+1, " is ", len(seen_files))
                seen_set = OurEvalDataset(
                file_list=seen_files,
                labels=seen_labels
            )
                seen_loader.append(DataLoader(
                seen_set,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=args.num_workers
            )      )
                
            return None, None, seen_loader, None, None



    return trn_loader, val_loader, None, None, class_weights

