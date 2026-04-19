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


from file_names.scenes_name import scenefiles

scene_to_speech = {
    "kitchen": "file_names.kitchen_files",
    "traffic": "file_names.traffic_files",
    "cafe": "file_names.cafe_files",
    "outdoor": "file_names.outdoor_files",
    "indoor": "file_names.indoor_files",
    "music": "file_names.music_files",
    "park": "file_names.park_files",
    "machine": "file_names.machine_files",
    "sport": "file_names.sport_files"
}

 
start = -5
end = 25


rows = []
def rms_norm(x, target=0.1):

    rms = torch.sqrt(torch.mean(x**2))
    return x * (target / (rms + 1e-8))



    
def load_env(file):

    data, sr = sf.read(file)

    if len(data) == 0:
        raise RuntimeError("Empty audio")

    wave = torch.from_numpy(data).float()

    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    else:
        wave = wave.T

    if wave.shape[0] > 1:
        wave = torch.mean(wave, dim=0, keepdim=True)

    return wave, sr, wave.shape[1]


def generate_snr_curve(length, sr, start_snr, end_snr, points=6):

    control = torch.rand(points) * (end_snr - start_snr) + start_snr

    idx = torch.linspace(0, length - 1, points)
    times = idx / sr

    control_t = control.view(1,1,-1)

    curve = F.interpolate(
        control_t,
        size=length,
        mode="linear",
        align_corners=True
    ).view(-1)

    return curve, control, times



out_dir = "fake/"

os.makedirs(out_dir, exist_ok=True)

csv_path = f"fake/fake_csv.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df = pd.DataFrame(rows)

if os.path.exists(csv_path):
    df.to_csv(csv_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_path, mode='w', header=True, index=False)

for scene, env_files in scenefiles.items():

    
    print(f"\nProcessing scene {scene}  ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧")
    scene_name = scene

    module = __import__(scene_to_speech[scene], fromlist=["dummy"])
    speech_list = getattr(module, f"{scene}_files")


    for speech_path in speech_list:

        speech_name = os.path.basename(speech_path)
        print(f"processing {speech_name} ✧｡٩(ˊᗜˋ )و✧*｡")
        
        speech_data, speech_sr = sf.read(speech_path)

        speech_wave = torch.from_numpy(speech_data).float()

        if speech_wave.ndim == 1:        # mono
            speech_wave = speech_wave.unsqueeze(0)

        else:                           # stereo
            speech_wave = speech_wave.T

        speech_len = speech_wave.shape[1]

        speech = rms_norm(speech_wave.squeeze())

        env_file = random.choice(env_files)
        print("env loading from", env_file)
        env_wave, env_sr, env_len = load_env(env_file)

        if env_sr != speech_sr:
            env_wave = torchaudio.transforms.Resample(env_sr, speech_sr)(env_wave)

        env_len = env_wave.shape[1]

        if env_len <= speech_len:
            start_idx = 0
        else:
            start_idx = random.randint(0, env_len - speech_len)

        env_segment = env_wave[:, start_idx:start_idx + speech_len]

        env_start = start_idx / speech_sr
        env_end = (start_idx + speech_len) / speech_sr

        noise = rms_norm(env_segment.squeeze())
        
        points = random.randint(5,10)
        snr_curve, snr_points, snr_times = generate_snr_curve(
            speech_len,
            speech_sr,
            start,
            end,
            points
        )

        speech_power = speech.pow(2).mean()
        noise_power = noise.pow(2).mean()

        snr_linear = 10 ** (snr_curve / 10)
        noise_scale = torch.sqrt(speech_power / (snr_linear * noise_power))

        mixed = speech + noise_scale * noise

        mixed = mixed / (mixed.abs().max() + 1e-8)

        mixed = mixed.unsqueeze(0)

        out_path = os.path.join(out_dir, f"fake_{speech_name}")
    
        torchaudio.save(out_path, mixed, speech_sr)

        for t, v in zip(snr_times.tolist(), snr_points.tolist()):

            rows.append({
                "speechfile": speech_name,
                "scene": scene_name,
                "envfilename": env_file,
                "envtimestart": env_start,
                "envtimeend": env_end,
                "snr_timestamp": float(t),
                "snr_value": float(v)
            })


df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)

print("Done づ ˘͈ ᵕ ˘͈ )づ")


