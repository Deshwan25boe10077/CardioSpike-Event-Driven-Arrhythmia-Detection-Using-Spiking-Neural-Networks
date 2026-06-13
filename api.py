from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# this shi killed me to fix
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 125, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# I just wanted to debug clearly so that's why the print 
print("Loading model...")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cardio_snn.pt', map_location=device))
model.eval()
print(" Model loaded boiii")

# routing the flask api
@app.route('/')
def serve_ui():
    """Serve HTML UI"""
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict arrhythmia from ECG signal"""
    data = request.json
    ecg_signal = np.array(data['ecg'], dtype=np.float32)
    
    # Normalize & pad to 500 samples, I'd increase it to more sample for better interpolation but my pc'd die
    ecg_padded = np.zeros(500, dtype=np.float32)
    ecg_padded[:min(len(ecg_signal), 500)] = ecg_signal[:500]
    
    # Add batch and channel dims: (500,) -> (1, 1, 500) batch and dims will be changes later
    x = torch.FloatTensor(ecg_padded).unsqueeze(0).unsqueeze(0).to(device)
    
    # Prediction snippet
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    
    normal_prob = float(probs[0])
    arrhythmia_prob = float(probs[1])
    
    return jsonify({
        'normal_prob': normal_prob,
        'arrhythmia_prob': arrhythmia_prob,
        'prediction': 'Normal' if normal_prob > 0.5 else 'Arrhythmia'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Starting your one night's sleep on http://127.0.0.1:5000")
    app.run(debug=False, port=5000, host='127.0.0.1')
