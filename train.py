import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# GENERATE CONTINUOUS 500-SAMPLE, because any more than that, I'd have to set a funeral for my pc ECG 
def generate_normal_ecg_continuous(num_samples=500, fs=250):
    """Generate continuous normal ECG (500 samples)"""
    t = np.arange(num_samples) / fs  # 0 to 2 seconds
    ecg = (
        0.3 * np.sin(2 * np.pi * 1.2 * t) +
        0.7 * np.sin(2 * np.pi * 4.5 * t) +
        0.25 * np.sin(2 * np.pi * 1.8 * t) +
        0.02 * np.random.randn(num_samples)
    )
    ecg = (ecg - ecg.min()) + 1
    return ecg

def generate_arrhythmic_ecg_continuous(num_samples=500, fs=250):
    """Generate continuous arrhythmic ECG (500 samples)"""
    t = np.arange(num_samples) / fs
    ecg = np.zeros(num_samples)
    
    for i in range(num_samples):
        ti = t[i]
        ampShift = 0.5 * np.sin(2 * np.pi * 0.5 * i / num_samples)
        
        if i % 150 == 0:  # Ectopic beat( add this to readme too)
            ecg[i] = (
                0.1 * np.sin(2 * np.pi * 3.2 * ti) +
                1.8 * np.sin(2 * np.pi * 7.5 * ti) +
                0.2 * np.sin(2 * np.pi * 2.8 * ti)
            )
        else:
            ecg[i] = (
                (0.3 + ampShift) * np.sin(2 * np.pi * (1.2 + 1.5*ampShift) * ti) +
                (0.8 + ampShift) * np.sin(2 * np.pi * (4.5 + 2*ampShift) * ti) +
                (0.4 + ampShift) * np.sin(2 * np.pi * (1.5 + ampShift) * ti) +
                (np.random.rand() - 0.5) * 0.3
            )
    
    ecg = (ecg - ecg.min()) + 1
    return ecg

print("=" * 50)
print("CardioSpike Training (Fixed)")
print("=" * 50)

print("\n[1/5] Generating continuous ECG data...")
normal_signals = np.array([generate_normal_ecg_continuous() for _ in range(150)])
arrhythmic_signals = np.array([generate_arrhythmic_ecg_continuous() for _ in range(150)])

# Check separability
normal_rms = np.sqrt(np.mean(normal_signals**2, axis=1))
arrhythmic_rms = np.sqrt(np.mean(arrhythmic_signals**2, axis=1))

print(f"Normal      RMS: {normal_rms.mean():.3f} ± {normal_rms.std():.3f}")
print(f"Arrhythmic  RMS: {arrhythmic_rms.mean():.3f} ± {arrhythmic_rms.std():.3f}")

X = np.vstack([normal_signals, arrhythmic_signals])
y = np.hstack([np.zeros(150, dtype=int), np.ones(150, dtype=int)])

idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

print(f"Total samples: {len(X)} (Normal: {(y==0).sum()}, Arrhythmic: {(y==1).sum()})")

#PREPARE DATA
print("\n[2/5] Preparing data...")
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# SIMPLE CNN coz complex sucks
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

print("\n[3/5] Training CNN...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_ds = TensorDataset(
    torch.FloatTensor(X_train).unsqueeze(1),
    torch.LongTensor(y_train)
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

best_val_acc = 0
for epoch in range(25):
    total_loss = 0
    correct = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(1) == y_batch).sum().item()
    
    acc = 100 * correct / len(X_train)
    
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val).unsqueeze(1).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)
        logits_val = model(X_val_t)
        val_acc = 100 * (logits_val.argmax(1) == y_val_t).sum().item() / len(y_val)
        best_val_acc = max(best_val_acc, val_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Train: {acc:.1f}% | Val: {val_acc:.1f}%")

print(f"\nBest validation accuracy: {best_val_acc:.1f}%")

torch.save(model.state_dict(), 'cardio_snn.pt')
print("\n[4/5] Model saved")
print("[5/5] Done. Run the damn app now with the api")
