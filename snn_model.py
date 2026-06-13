#this is snn v 3,  all the other version has problems with norse, so removed it completely in this v
import torch
import torch.nn as nn

class SimpleLIF(nn.Module):
    """Minimal Leaky Integrate-and-Fire neuron layer"""
    def __init__(self, input_size, output_size, tau=0.95, vth=1.0):
        super().__init__()
        self.weight = nn.Linear(input_size, output_size)
        self.tau = tau
        self.vth = vth
        self.v = None  # Initialize dynamically, static intialization showed a few problems 
    
    def forward(self, x):
        """
        x: (batch, input_size)
        returns: spikes (batch, output_size)
        """
        batch_size = x.shape[0]
        
        # Initialize state on first call, because can't start a call on second call
        if self.v is None or self.v.shape[0] != batch_size:
            self.v = torch.zeros(batch_size, self.weight.out_features, device=x.device)
        
        # Membrane potential: leaky decay + input, this is just heavy math from the paper
        self.v = self.tau * self.v + (1 - self.tau) * self.weight(x)
        
        # Spike if threshold crossed, hope it won't cross
        spikes = (self.v >= self.vth).float()
        
        # Reset after spike, for processing changes dynamically
        self.v = self.v * (1 - spikes)
        
        return spikes

class CardioSNN(nn.Module):
    def __init__(self, input_size=500, hidden1=64, hidden2=32, num_classes=2):
        super().__init__()
        self.lif1 = SimpleLIF(input_size, hidden1, tau=0.95, vth=1.0)
        self.lif2 = SimpleLIF(hidden1, hidden2, tau=0.95, vth=1.0)
        self.readout = nn.Linear(hidden2, num_classes)
    
    def forward(self, x):
        """
        x: (batch, time_steps, input_size), this sucked
        """
        batch_size, time_steps, _ = x.shape
        outputs = []
        
        for t in range(time_steps):
            # Process single timestep through both LIF layers, hidden layers ofc
            s1 = self.lif1(x[:, t])          # (batch, hidden1)
            s2 = self.lif2(s1)               # (batch, hidden2)
            outputs.append(s2)
        
        # Average spike counts over time
        spike_tensor = torch.stack(outputs, dim=1)  # (batch, time, hidden2)
        avg_spikes = spike_tensor.mean(dim=1)      # (batch, hidden2)
        
        # Classify
        logits = self.readout(avg_spikes)
        return logits

# Test
if __name__ == "__main__":
    model = CardioSNN(input_size=500, hidden1=64, hidden2=32, num_classes=2)
    x_dummy = torch.randn(4, 100, 500)
    out = model(x_dummy)
    print(f"Output shape: {out.shape}")
