## Real SceneFake

This repository contains baseline implementations for the **Real SceneFake** dataset.

### Evaluated Models

* **AASIST**
* **W2V2 + AASIST**
* **BEATs + AASIST**
* **RawTFNet**
* **XLSR-Mamba**

### Experiments:

1. **Generalization Compatibility (XLSR-Mamba)**
   Evaluates how well XLSR-Mamba trained on one dataset generalizes to another.

2. **K-shot Fine-tuning (XLSR-Mamba)**
   Measures performance when adapting the model with limited samples (few-shot setting).

3. **Cross-dataset Generalization**
   Tests robustness when training and testing across different datasets.

---

### Overall Cross-dataset Performance

![Cross-dataset Results](image.png)

---

## Authors

- Jiya Sinha, Aarthi S, Akshay Agarwal
- Email id: {jiya22, saarthi24, akagarwal}@iiserb.ac.in
- Affiliation: Trustworthy BiometraVision Lab, IISER Bhopal, India

---

## Dataset License and Usage Guidelines

- The dataset is to be used solely for its intended purpose i.e. SceneFake detection.

- Commercial use of the dataset is strictly prohibited.

- Any use of the dataset in research or academic work must include proper credit and citation of the repository and its associated paper, including in presentations and publications.

- Misuse of the voice data may result in legal consequences. Users must ensure the privacy and integrity of the dataset are maintained.

- The dataset owner reserves the right to revoke access at any time without prior notice.

