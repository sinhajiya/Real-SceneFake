### RawTFNet

Code adapted from:
[https://github.com/swagshaw/RawTFNet-Pytorch.git](https://github.com/swagshaw/RawTFNet-Pytorch.git)

#### Results (EER %)

| Training | SF (Utt) | Our Data (Utt) | Our Data (Seg) | Unseen (Utt) | Unseen (Seg) |
| -------- | -------: | -------------: | -------------: | -----------: | -----------: |
| SF       |    26.16 |          72.17 |          66.74 |        83.34 |        73.98 |
| Ours     |    48.36 |          16.52 |           8.80 |        15.22 |        22.23 |
| Combined |    10.88 |          11.30 |           7.54 |        11.59 |        16.79 |


#### Pretrained models available at:
https://huggingface.co/datasets/sinhajiya/Real_SceneFake/tree/main/pretrained_models/RawTFNet_models

### Citation

```bibtex
@inproceedings{xiao2025rawtfnet,
  title={RawTFNet: A Lightweight CNN Architecture for Speech Anti-spoofing},
  author={Xiao, Yang and Dang, Ting and Das, Rohan Kumar},
  booktitle={2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  pages={1997--2001},
  year={2025},
  organization={IEEE}
}
```