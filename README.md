CARDIO SPIKE - Arrhythmia Detection System
Version 1.0
================================================

OVERVIEW
--------
CardioSpike is a neural network-based ECG classification system for detecting 
cardiac arrhythmias. The system analyzes electrocardiogram signals and classifies 
them as either normal cardiac rhythm or abnormal (arrhythmia).

Technical: Convolutional Neural Network (CNN) on 1D ECG signals (500 samples).
No SNNs in current MVP - standard CNN for reliability and speed.


INSTALLATION
------------
Requirements:
  - Python 3.8+
  - PyTorch 2.0+
  - NumPy, SciPy, Flask

Setup:
  $ python -m venv venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt

Dependencies (requirements.txt):
  torch
  numpy
  scipy
  scikit-learn
  pandas
  matplotlib
  flask


QUICK START
-----------
1. Train the model:
   $ python train.py
   
   Output: cardio_snn.pt (trained weights)
   Expected: >95% validation accuracy

2. Start the API server:
   $ python api.py
   
   Output: Running on http://127.0.0.1:5000

3. Open web interface:
   http://localhost:5000
   
   - Click "Load Normal" or "Load Arrhythmic"
   - Click "ANALYZE SIGNAL"
   - View classification result


SYSTEM WORKFLOW
---------------
Input (ECG Signal, 500 samples @ 250 Hz)
  |
  v
Normalization & Padding
  |
  v
Convolutional Neural Network
  - Conv1d (7-kernel) + ReLU + MaxPool
  - Conv1d (5-kernel) + ReLU + MaxPool
  - Fully Connected (128 hidden) + ReLU
  - Output (2 classes: Normal, Arrhythmia)
  |
  v
Softmax Classification
  |
  v
Result (Class + Probability + Confidence)


DATA FORMATS
------------
ECG Input:
  - Duration: 2 seconds
  - Sampling: 250 Hz
  - Samples: 500
  - Units: millivolts (mV)

Model Output:
  - Normal Probability: 0.0 - 1.0
  - Arrhythmia Probability: 0.0 - 1.0
  - Classification: Normal or Arrhythmia
  - Confidence: Max(Normal, Arrhythmia) as %

Prediction Confidence:
  >90%  = High confidence (reliable)
  70-90% = Moderate confidence
  <70%  = Low confidence (uncertain)


FILE STRUCTURE
--------------
cardio-spike/
├── train.py           Main training script
├── api.py             Flask API server
├── index.html         Web interface
├── snn_model.py       Model definition
├── data_loader.py     Synthetic data generator
├── cardio_snn.pt      Trained model (generated)
└── requirements.txt   Python dependencies


TRAINING
--------
Dataset:
  - Synthetic ECG signals (no external data needed)
  - 150 normal beats + 150 arrhythmic beats
  - Auto-generated during training

Training Parameters:
  - Epochs: 25
  - Batch Size: 16
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Loss: Cross-Entropy

Expected Results:
  - Training Accuracy: >99%
  - Validation Accuracy: >95%
  - Training Time: ~2-3 minutes (CPU)


API ENDPOINTS
-------------
GET /health
  Returns: {"status": "ok"}
  Usage: System health check

POST /predict
  Input: {"ecg": [list of 500 float values]}
  Returns: {
    "prediction": "Normal" or "Arrhythmia",
    "normal_prob": float (0.0-1.0),
    "arrhythmia_prob": float (0.0-1.0)
  }


WEB INTERFACE
-------------
Left Panel (ECG Signal Input):
  - Load Normal: Generate normal ECG signal
  - Load Arrhythmic: Generate irregular ECG signal
  - Clear: Reset data
  - Graph: Visual ECG waveform display
  - Data Table: Signal statistics (min, max, range)
  - Analyze Signal: Run classification

Right Panel (Results):
  - Classification: Predicted class
  - Probabilities: Normal and Arrhythmia %
  - Processing Time: Inference latency (ms)
  - Model Confidence: Overall decision confidence
  - Status Box: Color-coded result
    * Green = Normal Rhythm
    * Red = Arrhythmia Detected
  - System Status: Last analysis timestamp


INTERPRETATION
--------------
Normal Rhythm:
  - Regular, predictable ECG pattern
  - P-wave → QRS complex → T-wave sequence
  - Consistent timing and amplitude

Arrhythmia Detected:
  - Irregular heartbeat
  - Ectopic (premature) beats
  - Variable RR intervals
  - Abnormal waveform morphology

Confidence Interpretation:
  High Confidence (>90%):
    - Rely on classification
    - Safe for clinical decision support

Moderate Confidence (70-90%):
    - Use as supporting evidence only
    - Recommend secondary review

Low Confidence (<70%):
    - Borderline case
    - Manual review recommended


TECHNICAL NOTES FOR EXPERTS
----------------------------
Model Architecture:
  - Input: (1, 500) - single channel ECG
  - Conv1: 32 filters, kernel=7, padding=3
  - MaxPool: stride=2 (output: 32x250)
  - Conv2: 64 filters, kernel=5, padding=2
  - MaxPool: stride=2 (output: 64x125)
  - FC1: 64*125 → 128 (ReLU)
  - FC2: 128 → 2 (softmax)

Training Details:
  - Data augmentation: None (synthetic data)
  - Normalization: Min-max per signal
  - Activation: ReLU (hidden), Softmax (output)
  - Regularization: None
  - Dropout: None

Synthetic Data Generation:
  Normal ECG:
    - 0.3*sin(2π*1.2*t) [P-wave]
    - 0.7*sin(2π*4.5*t) [QRS]
    - 0.25*sin(2π*1.8*t) [T-wave]
    - Gaussian noise: σ=0.02

  Arrhythmic ECG:
    - Variable frequency shifts (±0.5)
    - Ectopic beats every 150 samples
    - Increased noise (σ=0.3)
    - Amplitude variation


LIMITATIONS
-----------
- Trained on synthetic ECG data only
- No real patient data validation
- Single channel analysis (lead II equivalent)
- Binary classification (Normal vs Arrhythmia)
- Does not distinguish arrhythmia subtypes
- Not clinically validated
- Research/demonstration only


FUTURE IMPROVEMENTS
-------------------
- Real MIT-BIH dataset training
- Multi-class classification (AFib, VT, PVC, etc)
- Multi-lead ECG support
- Spiking Neural Network implementation
- Neuromorphic hardware deployment (Intel Loihi)
- Continuous monitoring capability
- Patient-specific baseline adaptation


REFERENCES
----------
MIT-BIH Arrhythmia Database:
  https://physionet.org/content/mitdb/

ECG Signal Processing:
  Pan, J., & Tompkins, W. J. (1985).
  "A Real-Time QRS Detection Algorithm"

Spiking Neural Networks:
  Maass, W. (1997). "Networks of Spiking Neurons"


CONTACT & SUPPORT
-----------------
System Status: Operational
Last Updated: June 2026
Version: 1.0 MVP


DISCLAIMER
----------
This system is for research and demonstration purposes only.
Not intended for clinical diagnosis or patient care.
Always consult qualified medical professionals for ECG interpretation.
