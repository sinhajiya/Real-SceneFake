
import os
import random
import torch
import torchaudio
import pandas as pd
import torch.nn.functional as F
import subprocess
import warnings
warnings.filterwarnings("ignore", module="torchaudio")
import soundfile as sf
# from speechbrain.augment.time_domain import SpeedPerturb


def add_speech_per(wave, sr=16000, target_len=None):
    from speechbrain.augment.time_domain import SpeedPerturb

    wave = wave.squeeze()
    if wave.dim() == 0 or wave.numel() < 100:
        return wave

    try:
        perturbator = SpeedPerturb(orig_freq=sr, speeds=[90, 80, 110, 120])
        out = perturbator(wave.unsqueeze(0))
        if not isinstance(out, torch.Tensor):
            return wave
        out = out.squeeze()
        if out.dim() != 1 or out.numel() < 100:
            return wave

    except Exception:
        return wave

    if target_len is not None:
        if out.shape[0] > target_len:
            out = out[:target_len]
        else:
            out = F.pad(out, (0, target_len - out.shape[0]))

    return out


def add_noise_floor(x):
    floor_db = random.uniform(-65, -55)
    signal_rms = torch.sqrt(torch.mean(x**2))
    floor_rms = signal_rms * (10 ** (floor_db / 20))
    noise = torch.randn_like(x)
    noise = noise * (floor_rms / (torch.sqrt(torch.mean(noise**2)) + 1e-8))
    return x + noise



def rms(x):
    return torch.sqrt(torch.mean(x**2))


def dbfs_to_linear(db):
    return 10 ** (db / 20)

def loudness_adjust_file(wave, min_db=-35, max_db=-15):

    if wave.dim() > 1:
        wave = torch.mean(wave, dim=0)

    x = wave.squeeze()

    current_rms = rms(x)
    target_db = random.uniform(min_db, max_db)
    target_rms = dbfs_to_linear(target_db)

    scale = target_rms / (current_rms + 1e-8)
    y = x * scale

    peak = y.abs().max()
    if peak > 1:
        y = y / peak

    return y  # 🔴 NO unsqueeze


def loudness_adjust_fikoinle(wave, min_db=-35, max_db=-15):

    # wave, sr = torchaudio.load(in_path)

    if wave.shape[0] > 1:
        wave = torch.mean(wave, dim=0, keepdim=True)

    x = wave.squeeze()

    current_rms = rms(x)

    target_db = random.uniform(min_db, max_db)

    target_rms = dbfs_to_linear(target_db)

    scale = target_rms / (current_rms + 1e-8)
    y = x * scale

    peak = y.abs().max()
    if peak > 1:
        y = y / peak

    y = y.unsqueeze(0)

    return y



def mp3_compression(waveform_ndarray, sr):
    from audiomentations import Mp3Compression

    transform = Mp3Compression(
        min_bitrate=16,
        max_bitrate=96,
        backend="fast-mp3-augment",
        preserve_delay=False,
        p=1.0
    )

    return transform(waveform_ndarray, sample_rate=sr)


from audiomentations import (
    Compose,
    AddGaussianSNR,
    Mp3Compression,
    Aliasing,
    BandPassFilter,
    Gain,
    GainTransition,
    LoudnessNormalization
)
import numpy as np

def audioaugment(waveform, sr, augmentations=[]):
    ''''
    augmentations is a list of augmentations that can be applied. the allowed names are: "gaussian_noise_snr", "mp3_compression", "aliasing", "band_pass_filter", "gain", "gain_transitions", "loudness_norm", "speech_pertubation"

    
    ## definitions from the audioaugmentations library:
    1. Gaussian Noise SNR: The AddGaussianSNR transform injects Gaussian noise into an audio signal. It applies a Signal-to-Noise Ratio (SNR) that is chosen randomly from a uniform distribution on the Decibel scale. This choice is consistent with the nature of human hearing, which is logarithmic rather than linear.

    2. Mp3 comressions: Compress the audio using an MP3 encoder to lower the audio quality. This may help machine learning models deal with compressed, low-quality audio.

    3. Aliasing: Downsample the audio to a lower sample rate by linear interpolation, without low-pass filtering it first, resulting in aliasing artifacts. You get aliasing artifacts when there is high-frequency audio in the input audio that falls above the Nyquist frequency of the chosen target sample rate. Audio with frequencies above the Nyquist frequency cannot be reproduced accurately and gets "reflected"/mirrored to other frequencies. The aliasing artifacts replace the original high-frequency signals. The result can be described as coarse and metallic.

    4. BandPassFilter: Here we input a high-quality speech recording and apply BandPassFilter with a center frequency of 2500 Hz and a bandwidth fraction of 0.8, which means that the bandwidth in this example is 2000 Hz, so the low frequency cutoff is 1500 Hz and the high frequency cutoff is 3500 Hz. One can see in the spectrogram that the high and the low frequencies are both attenuated in the output. If you listen to the audio example, you might notice that the transformed output almost sounds like a phone call from the time when phone audio was narrowband and mostly contained frequencies between ~300 and ~3400 Hz.

    5. Gain: Multiply the audio by a random amplitude factor to reduce or increase the volume. This technique can help a model become somewhat invariant to the overall gain of the input audio.

    6. Gain Transitions: Gradually change the volume up or down over a random time span. Also known as fade in and fade out. The fade works on a logarithmic scale, which is natural to human hearing. 
    
    7. Loudness norm: Apply a constant amount of gain to match a specific loudness (in LUFS). This is an implementation of ITU-R BS.1770-4.

    8. Speed Perturbation: With Speed perturbation, we resample the audio signal to a sampling rate that is a bit different from the original one. With this simple trick we can synthesize a speech signal that sounds a bit “faster” or “slower” than the original one. Note that not only the speaking rate is affected, but also the speaker characteristics such as pitch and formants 

    '''
    # print("Applying augmentations:", augmentations if augmentations else "None")

    if len(augmentations)== 0:
        return waveform

    transform_list = []

    for aug in augmentations:

        if aug == "gaussian_noise_snr":
            transform_list.append(
                AddGaussianSNR(
                    min_snr_db=5,
                    max_snr_db=40,
                    p=1
                )
            )



        elif aug == "aliasing":
            transform_list.append(
                Aliasing(
                    min_sample_rate=2000,
                    max_sample_rate=8000,
                    p=1
                )
            )

        elif aug == "band_pass_filter":
            transform_list.append(
                BandPassFilter(
                    min_center_freq=300,
                    max_center_freq=3000,
                    min_bandwidth_fraction=0.5,
                    max_bandwidth_fraction=1.5,
                    p=1
                )
            )

        elif aug == "gain":
            transform_list.append(
                Gain(
                    min_gain_db=-12,
                    max_gain_db=12,
                    p=1
                )
            )

        elif aug == "gain_transitions":
            transform_list.append(
                GainTransition(
                    min_gain_db=-24,
                    max_gain_db=6,
                    min_duration=0.2,
                    max_duration=2.0,
                    p=1
                )
            )

        elif aug == "loudness_norm":
            transform_list.append(
                LoudnessNormalization(max_lufs = -15, min_lufs = -35, p=1)
                                  
            
                                  )
            
        elif aug == "mp3_compression":
            transform_list.append(
                Mp3Compression(
                    min_bitrate=32,
                    max_bitrate=128,
                    p=1
                )
            )
            
            
        # elif aug == "speech_pertubation":
        #     transform_list.append(
        #         add_speech_per()
                                  
        #                           )
            
        else:
            raise ValueError(f"Unknown augmentation: {aug}")

    augment = Compose(transform_list)

    return augment(samples=waveform, sample_rate=sr)

def audioaugment_double(waveform, sr, augmentations):
    

    if not augmentations:
        return [waveform]

    augmented = audioaugment(waveform, sr, augmentations)

    return [waveform, augmented]
