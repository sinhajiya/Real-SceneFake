## AASIST


### Training 

To train the model:
bash
```
python main.py --config <path to conf file>
````
To evaluate:
bash
```
python main.py --config  <path to conf file> --eval 
````

To perform scenefake eval set scenefake-eval to true in conf file.

---
Code adapted from:

* [https://github.com/clovaai/aasist.git](https://github.com/clovaai/aasist.git)
* [https://github.com/apple-yinhan/EnvSDD.git](https://github.com/apple-yinhan/EnvSDD.git)

This directory supports three models:

* **AASIST**
* **W2V2 + AASIST**
* **BEATs + AASIST**

---

Pretrained models available at:  https://huggingface.co/datasets/sinhajiya/Real_SceneFake/tree/main/pretrained_models

## Results (EER %)

### AASIST

| Training | SF (Utt) | Seen (Utt) | Seen (Seg) | Unseen (Utt) | Unseen (Seg) |
| -------- | -------: | ---------: | ---------: | -----------: | -----------: |
| SF       |    17.32 |      70.43 |      72.46 |        80.43 |        74.72 |
| Ours     |    50.18 |       3.48 |       5.48 |        10.87 |        13.38 |
| Combined |    23.99 |       5.22 |       6.40 |        18.84 |        20.48 |

---

### W2V2 + AASIST

| Training |  SF (Utt) | Seen (Utt) | Seen (Seg) | Unseen (Utt) | Unseen (Seg) |
| -------- | --------: | ---------: | ---------: | -----------: | -----------: |
| SF       |     13.62 |      62.61 |      55.66 |        76.81 |        70.39 |
| Ours     | **50.97** |   **4.35** |   **4.80** |     **4.35** |    **11.44** |
| Combined | **15.09** |   **0.87** |   **2.97** |    **11.59** |    **16.33** |

---

### BEATs + AASIST

| Training | SF (Utt) | Seen (Utt) | Seen (Seg) | Unseen (Utt) | Unseen (Seg) |
| -------- | -------: | ---------: | ---------: | -----------: | -----------: |
| SF       |    24.40 |      50.43 |      50.63 |        65.94 |        53.50 |
| Ours     |    50.00 |       5.22 |       5.94 |         2.90 |         7.47 |
| Combined |    23.71 |      13.04 |       6.63 |        13.77 |        21.86 |

---


## Citation

```bibtex
@inproceedings{jung2022aasist,
  title={AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6367--6371},
  year={2022},
  organization={IEEE}
}
```
