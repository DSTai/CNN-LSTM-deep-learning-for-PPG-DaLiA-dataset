# CNN-LSTM Deep Learning for PPG-DaLiA Dataset

A deep learning project that applies a **hybrid CNN–LSTM architecture** to improve heart rate estimation using the **PPG-DaLiA wearable dataset**.

This project was developed as part of my **AI pre-thesis research**, focusing on physiological signal processing and deep learning for wearable health monitoring.

---

# Project Overview

Photoplethysmography (PPG) signals are widely used in wearable devices for **heart rate monitoring**. However, signals collected in real-world environments often contain **motion artifacts and noise**, which makes accurate prediction challenging.

This project investigates whether a **CNN-LSTM hybrid deep learning model** can improve prediction accuracy by combining:

- **CNN** → automatic feature extraction from physiological signals  
- **LSTM** → temporal dependency learning from sequential data  

The model was evaluated on the **PPG-DaLiA dataset**, a widely used benchmark dataset for wearable heart rate estimation.

---

# Model Architecture

The proposed hybrid model combines:

### Convolutional Neural Networks (CNN)

- Extract spatial features from PPG and accelerometer signals

### Long Short-Term Memory (LSTM)

- Capture temporal dependencies in physiological signals

### Fully Connected Layers

- Perform final prediction

Pipeline:

---
PPG + Accelerometer Signals

↓

CNN Feature Extraction

↓

LSTM Temporal Modeling

↓

Dense Layers

↓

Heart Rate Prediction


---
# Dataset

This project uses the **PPG-DaLiA dataset**, which contains synchronized physiological signals collected from wearable sensors during daily activities.

Dataset characteristics:

- Wrist **PPG signals**
- Chest & wrist **accelerometer signals**
- Ground truth **heart rate**
- Multiple **activity scenarios**

Reference:

Reiss, Attila; Indlekofer, Ina; Schmidt, Philip.  
PPG-DaLiA Dataset — UCI Machine Learning Repository  
https://doi.org/10.24432/C53890

Related paper:

Reiss A., Indlekofer I., Schmidt P., Van Laerhoven K.  
Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks.  
Sensors, 2019.

---

# Experiments

Three sensor configurations were tested:

| Experiment | Signals Used |
|---|---|
| Chest Accelerometer | Chest motion signals |
| Wrist Accelerometer | Wrist motion signals |
| Chest + Wrist Accelerometer | Combined motion signals |

Two models were compared:

- **LSTM baseline**
- **Proposed CNN-LSTM hybrid model**

---

# Results

## Average Accuracy Comparison

| Sensor Configuration | LSTM | CNN-LSTM | Improvement |
|---|---|---|---|
| Chest Accelerometer | 0.6267 | **0.6698** | **+6.88%** |
| Wrist Accelerometer | 0.5587 | **0.6460** | **+15.63%** |
| Chest + Wrist Accelerometer | 0.4827 | **0.6427** | **+33.15%** |

The hybrid **CNN-LSTM model consistently outperforms the LSTM baseline** across all sensor configurations.

The largest improvement occurs when **combining chest and wrist accelerometer signals**, suggesting that **multimodal motion data improves prediction performance**.

---

# Subject-Level Results

## Chest Accelerometer

| Subject | LSTM Accuracy | CNN-LSTM Accuracy |
|---|---|---|
| S1 | 0.50 | 0.56 |
| S2 | 0.62 | 0.64 |
| S3 | 0.59 | 0.67 |
| S4 | 0.62 | 0.68 |
| S5 | 0.58 | 0.69 |
| S6 | 0.75 | 0.82 |
| S7 | 0.70 | 0.77 |
| S8 | 0.73 | 0.78 |
| S9 | 0.73 | 0.81 |
| S10 | 0.50 | 0.63 |
| S11 | 0.68 | 0.78 |
| S12 | 0.49 | 0.62 |
| S13 | 0.65 | 0.85 |
| S14 | 0.60 | 0.64 |
| S15 | 0.66 | 0.71 |

---

## Wrist Accelerometer

| Subject | LSTM Accuracy | CNN-LSTM Accuracy |
|---|---|---|
| S1 | 0.63 | 0.70 |
| S2 | 0.53 | 0.58 |
| S3 | 0.54 | 0.65 |
| S4 | 0.61 | 0.62 |
| S5 | 0.33 | 0.43 |
| S6 | 0.62 | 0.79 |
| S7 | 0.60 | 0.74 |
| S8 | 0.29 | 0.40 |
| S9 | 0.61 | 0.71 |
| S10 | 0.58 | 0.63 |
| S11 | 0.69 | 0.75 |
| S12 | 0.49 | 0.61 |
| S13 | 0.67 | 0.75 |
| S14 | 0.62 | 0.68 |
| S15 | 0.57 | 0.65 |

---

## Chest + Wrist Accelerometer

| Subject | LSTM Accuracy | CNN-LSTM Accuracy |
|---|---|---|
| S1 | 0.35 | 0.48 |
| S2 | 0.55 | 0.70 |
| S3 | 0.15 | 0.44 |
| S4 | 0.42 | 0.54 |
| S5 | 0.28 | 0.74 |
| S6 | 0.49 | 0.53 |
| S7 | 0.73 | 0.79 |
| S8 | 0.63 | 0.77 |
| S9 | 0.62 | 0.69 |
| S10 | 0.41 | 0.61 |
| S11 | 0.49 | 0.61 |
| S12 | 0.44 | 0.50 |
| S13 | 0.74 | 0.90 |
| S14 | 0.40 | 0.64 |
| S15 | 0.54 | 0.70 |

---

# Repository Structure
```
CNN-LSTM-deep-learning-for-PPG-DaLiA-dataset
│
├── hybrid/ # CNN-LSTM hybrid models
├── weights/ # Trained model weights
├── reports/ # Experimental results
│
├── dalia-activity-chest-acc.py
├── dalia-activity-wrist-acc.py
├── dalia-activity-chest-wrist-acc.py
│
├── dalia-activity-chest-acc-hybridmodel.py
├── dalia-activity-wrist-acc-hybridmodel.py
├── dalia-activity-chest-wrist-acc-hybridmodel.py
│
├── PPG_FieldStudy_readme.pdf
└── README.md


```

# How to Run

Clone the repository
```
git clone https://github.com/DSTai/CNN-LSTM-deep-learning-for-PPG-DaLiA-dataset.git
```
Install dependencies
```
pip install numpy pandas tensorflow keras scikit-learn
```
Run an experiment
```
python dalia-activity-wrist-acc-hybridmodel.py
```
# Applications

This research can contribute to:

- Wearable health monitoring
- Fitness tracking devices
- Real-time physiological signal analysis
- AI-powered healthcare systems

---

# Future Work

Possible improvements:

- Transformer-based time-series models
- Motion artifact removal techniques
- Multimodal fusion (PPG + ECG + IMU)
- Real-time deployment for wearable devices

---

# Author

**THD Tran**

AI / Machine Learning Research Enthusiast  
Deep Learning for Physiological Signal Processing

GitHub  
https://github.com/DSTai

---

# Citation

If you use this project, please cite:

Reiss A., Indlekofer I., Schmidt P., Van Laerhoven K.  
Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks.  
Sensors, 2019.