import os
import random
import pandas as pd
from itertools import combinations

FAKE_DIR = "/home/bs_thesis/Codes/FakeCreation/Ours/fake_no_time_varying"
REAL_DIR = "/home/bs_thesis/datasets/OurDataset/SceneFake-Wild-Real"
META_FILE = "/home/bs_thesis/datasets/OurDataset/SceneFakeAudio_info.csv"
OUT_DIR = "/home/bs_thesis/datasets/OurDataset/Ours_Fake/protocols/notimevaryingsnr"


os.makedirs(OUT_DIR, exist_ok=True)

AUTO_SELECT_SCENES = False
UNSEEN_SCENES = {'s04', 's07'} 

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SEED = 42
random.seed(SEED)

meta = pd.read_csv(META_FILE)


meta = meta.rename(columns={
    "File name": "file",
    "Speaker Id": "speaker",
    "Scene": "scene"
})
print("columns ", meta.columns)
required = {"file", "speaker", "scene"}
missing = required - set(meta.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"\nTotal samples: {len(meta)}")
print(f"Total speakers: {meta['speaker'].nunique()}")

multi_scene = meta.groupby("speaker")["scene"].nunique()
multi_scene = multi_scene[multi_scene > 1]
print(f"Speakers in multiple scenes: {len(multi_scene)}")

if AUTO_SELECT_SCENES:
    all_scenes = meta["scene"].unique()
    best_choice, min_overlap = None, float("inf")

    for combo in combinations(all_scenes, 2):
        unseen = meta[meta["scene"].isin(combo)]
        overlap = len(set(unseen["speaker"]) &
                      set(meta[~meta["scene"].isin(combo)]["speaker"]))

        if overlap < min_overlap:
            min_overlap = overlap
            best_choice = combo

    UNSEEN_SCENES = set(best_choice)
    print(f"\n[Auto] Unseen scenes: {UNSEEN_SCENES} (overlap={min_overlap})")

unseen_meta = meta[meta["scene"].isin(UNSEEN_SCENES)]
blocked_speakers = set(unseen_meta["speaker"])

remaining_meta = meta[
    (~meta["scene"].isin(UNSEEN_SCENES)) &
    (~meta["speaker"].isin(blocked_speakers))
]

print(f"Unseen samples: {len(unseen_meta)}")
print(f"Remaining samples: {len(remaining_meta)}")

speaker_counts = (
    remaining_meta.groupby("speaker")
    .size()
    .sort_values(ascending=False)
)

train_spk, val_spk, test_spk = set(), set(), set()
train_count = val_count = test_count = 0
total_samples = len(remaining_meta)

for spk, count in speaker_counts.items():

    train_ratio_now = train_count / total_samples
    val_ratio_now = val_count / total_samples

    if train_ratio_now < TRAIN_RATIO:
        train_spk.add(spk)
        train_count += count

    elif val_ratio_now < VAL_RATIO:
        val_spk.add(spk)
        val_count += count

    else:
        test_spk.add(spk)
        test_count += count

print(f"Train: {train_count}")
print(f"Val: {val_count}")
print(f"Seen-test: {test_count}")

train_meta = remaining_meta[remaining_meta["speaker"].isin(train_spk)]
val_meta = remaining_meta[remaining_meta["speaker"].isin(val_spk)]
seen_test_meta = remaining_meta[remaining_meta["speaker"].isin(test_spk)]


def assert_no_overlap(a, b, name):
    if set(a) & set(b):
        raise ValueError(f"Leakage: {name}")

assert_no_overlap(train_spk, val_spk, "train-val")
assert_no_overlap(train_spk, test_spk, "train-test")
assert_no_overlap(val_spk, test_spk, "val-test")
assert_no_overlap(blocked_speakers, train_spk, "unseen-train")
assert_no_overlap(blocked_speakers, val_spk, "unseen-val")
assert_no_overlap(blocked_speakers, test_spk, "unseen-test")

print("✔ No speaker leakage")


total = len(meta)
used = len(train_meta) + len(val_meta) + len(seen_test_meta) + len(unseen_meta)


print(f"Train samples: {len(train_meta)}")
print(f"Val samples: {len(val_meta)}")
print(f"Seen-test samples: {len(seen_test_meta)}")
print(f"Unseen-test samples: {len(unseen_meta)}")

print(f"Original: {total}")
print(f"Used: {used}")
print(f"Dropped: {total - used}")


def write_protocol(df, outfile, name):
    lines = []
    missing_fake = []

    for _, row in df.iterrows():
        real = row["file"]
        fake = f"fake_{real}"

        real_path = os.path.join(REAL_DIR, real)
        fake_path = os.path.join(FAKE_DIR, fake)

        if os.path.exists(real_path):
            lines.append(f"{real_path} 0")
        if os.path.exists(fake_path):
            lines.append(f"{fake_path} 1")
        else:
            missing_fake.append(fake)

    random.shuffle(lines) 
    with open(outfile, "w") as f:
        f.write("\n".join(lines))

    print(f"\n[{name}] {len(lines)} entries")

    if missing_fake:
        print(f"[{name}] Missing fake: {len(missing_fake)}")

write_protocol(train_meta, f"{OUT_DIR}/train.txt", "TRAIN")
write_protocol(val_meta, f"{OUT_DIR}/val.txt", "VAL")
write_protocol(seen_test_meta, f"{OUT_DIR}/seen_test.txt", "SEEN_TEST")
write_protocol(unseen_meta, f"{OUT_DIR}/unseen_test.txt", "UNSEEN_TEST")

print("\nDone.")