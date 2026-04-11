## Real SceneFake

This repository contains baseline implementations for the **Real SceneFake** dataset.


---

### Authors

- Jiya Sinha, Aarthi S, Akshay Agarwal
- Email id: {jiya22, saarthi24, akagarwal}@iiserb.ac.in
- Affiliation: Trustworthy BiometraVision Lab, IISER Bhopal, India

----


### Evaluated Models

* **AASIST**
* **W2V2 + AASIST**
* **BEATs + AASIST**
* **RawTFNet**
* **XLSR-Mamba**



Pretrained models are available at: https://huggingface.co/datasets/sinhajiya/Real_SceneFake

--- 

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

### Dataset License and Usage Guidelines


#### 1. Intended Use
This dataset may be used solely for research and development of audio deepfake detection systems. Any use outside this scope is strictly prohibited.

#### 2. Non-Commercial Use
Commercial use of the dataset is strictly prohibited.

#### 3. Attribution and Citation
Any use of the dataset in research or academic work must include proper credit and citation of the repository and its associated paper, including in presentations and publications.

#### 4. Misuse and Responsibility
Misuse of the voice data may result in legal consequences. Users must ensure the privacy and integrity of the dataset are maintained.
Users must not:

- misuse the voice data for impersonation, synthesis, or harmful applications 
- attempt to identify individuals or violate privacy

Users are fully responsible for ensuring ethical use and compliance with applicable laws.
Misuse may result in legal action.
#### 5. Access and Revocation
The dataset owner reserves the right to revoke access at any time without prior notice.

