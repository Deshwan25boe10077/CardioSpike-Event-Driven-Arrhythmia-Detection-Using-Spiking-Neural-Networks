# ❤️ CardioSpike
### Event-Driven Arrhythmia Detection Using Spiking Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![snnTorch](https://img.shields.io/badge/snnTorch-SNN_Framework-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## 📖 Abstract

Cardiovascular diseases remain one of the leading causes of mortality worldwide, making early and accurate detection of cardiac abnormalities essential. Traditional deep learning approaches for Electrocardiogram (ECG) analysis often require substantial computational resources and energy consumption, limiting their deployment on wearable and edge devices.

**CardioSpike** introduces an event-driven arrhythmia detection framework based on **Spiking Neural Networks (SNNs)**, which mimic the information processing mechanism of biological neurons. By converting ECG signals into spike trains and leveraging temporal dynamics, CardioSpike aims to achieve accurate arrhythmia classification while significantly reducing computational overhead and power consumption.

This project explores the application of neuromorphic computing principles to healthcare, enabling intelligent and energy-efficient cardiac monitoring systems suitable for real-time deployment.

---

## 🎯 Problem Statement

Current ECG classification systems based on Artificial Neural Networks (ANNs) provide high accuracy but often suffer from:

- High energy consumption
- Large memory requirements
- Limited suitability for edge devices
- Reduced biological plausibility

CardioSpike addresses these challenges through:

- Event-driven computation
- Sparse neural activity
- Temporal information encoding
- Energy-efficient inference

---

## 🚀 Key Features

- ECG signal preprocessing pipeline
- Spike encoding of continuous ECG waveforms
- Spiking Neural Network architecture
- Multi-class arrhythmia classification
- Event-driven inference
- Model training and evaluation framework
- Performance visualization
- Research-oriented modular architecture
- Neuromorphic computing compatibility
- Edge AI deployment potential

---

## 🧠 Why Spiking Neural Networks?

Unlike conventional neural networks that process continuous values, SNNs communicate through discrete spikes.

### Advantages

✅ Lower energy consumption

✅ Temporal pattern recognition

✅ Sparse computation

✅ Biological plausibility

✅ Neuromorphic hardware compatibility

These characteristics make SNNs highly suitable for wearable healthcare applications where power efficiency is critical.

---

## 🏗️ System Architecture

•	This is how our model works on converting the ECG signals to spike trains so that we can overcome the complexities of the standard ECG 

•	Since ECG signals are defined by distinct temporal shapes (like the P-wave, QRS complex, and T-wave), the goal of an event encoder is to capture these precise     amplitude changes over time while discarding redundant, static data.


                   ECG Signal
                        │
                        ▼
              Signal Preprocessing
                        │
                        ▼
                 Spike Encoding
                        │
                        ▼
             Spiking Neural Network
                        │
                        ▼
               Feature Extraction
                        │
                        ▼
              Arrhythmia Classifier
                        │
                        ▼
                Predicted Class

📊 Dataset

This project can be trained using publicly available ECG datasets.

MIT-BIH Arrhythmia Database
48 annotated ECG recordings
Sampling Frequency: 360 Hz
Widely used benchmark dataset
PTB Diagnostic ECG Database
Diagnostic ECG recordings
Multiple cardiac conditions
Suitable for classification tasks
PhysioNet Databases

Provides several clinically validated ECG datasets.
