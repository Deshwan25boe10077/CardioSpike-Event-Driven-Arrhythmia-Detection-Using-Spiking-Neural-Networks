# CardioSpike: Event-Driven Arrhythmia Detection Using Spiking Neural Networks

## Overview

CardioSpike is a research-oriented project that leverages **Spiking Neural Networks (SNNs)** for efficient and biologically inspired detection of cardiac arrhythmias from Electrocardiogram (ECG) signals. Unlike traditional Artificial Neural Networks (ANNs), SNNs process information through discrete spikes, enabling event-driven computation with significantly lower energy consumption and improved suitability for edge and neuromorphic hardware.

The project aims to classify ECG recordings into normal and abnormal heart rhythms while maintaining high accuracy and computational efficiency.

---

## Features

- Event-driven ECG signal processing
- ECG-to-spike encoding techniques
- Spiking Neural Network architecture for arrhythmia classification
- Support for multiple arrhythmia classes
- Energy-efficient inference
- Training and evaluation pipeline
- Performance visualization and analysis tools
- Compatible with neuromorphic computing research workflows

---

## Project Objectives

1. Preprocess raw ECG signals.
2. Convert continuous ECG waveforms into spike trains.
3. Train Spiking Neural Networks for arrhythmia classification.
4. Evaluate model performance using standard medical metrics.
5. Compare SNN performance with conventional deep learning approaches.
6. Investigate energy efficiency benefits of event-driven computing.

---

## Dataset

The project can be trained and evaluated using publicly available ECG datasets such as:

- MIT-BIH Arrhythmia Database
- PTB Diagnostic ECG Database
- PhysioNet ECG Collections

### Example Classes

- Normal Sinus Rhythm (NSR)
- Premature Ventricular Contractions (PVC)
- Atrial Fibrillation (AF)
- Ventricular Tachycardia (VT)
- Other clinically relevant arrhythmias

---

## System Architecture

```text
Raw ECG Signal
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
 Classification Layer
       │
       ▼
 Arrhythmia Prediction
