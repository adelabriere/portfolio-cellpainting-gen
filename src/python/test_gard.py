import torch
import torch.nn as nn
import numpy as np


# TO DELETE A SIMPLE TEST TO CHECK THAT BOTH APPROAHCES ARE EQUIVALENT

# Set seed for reproducibility
torch.manual_seed(42)

# Simple discriminator model
class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Test function
def test_wgan_gradient_equivalence():
    # Create model and inputs
    D = SimpleDiscriminator()
    real_images = torch.randn(5, 10)
    fake_images = torch.randn(5, 10)
    
    # Approach 1: Separate backward calls with one/mone
    D1 = SimpleDiscriminator()
    D1.load_state_dict(D.state_dict())  # Same initial weights
    
    one = torch.tensor(1.0)
    mone = torch.tensor(-1.0)
    
    # Forward pass
    d_real_1 = D1(real_images).mean()
    d_fake_1 = D1(fake_images).mean()
    
    # Backward pass - approach 1
    D1.zero_grad()
    d_real_1.backward(mone)  # With -1
    d_fake_1.backward(one)   # With +1
    
    # Store gradients from approach 1
    grads_approach1 = [p.grad.clone() for p in D1.parameters()]
    
    # Approach 2: Combined loss
    D2 = SimpleDiscriminator()
    D2.load_state_dict(D.state_dict())  # Same initial weights
    
    # Forward pass
    d_real_2 = D2(real_images).mean()
    d_fake_2 = D2(fake_images).mean()
    
    # Backward pass - approach 2
    D2.zero_grad()
    d_loss = d_fake_2 - d_real_2
    d_loss.backward()
    
    # Store gradients from approach 2
    grads_approach2 = [p.grad.clone() for p in D2.parameters()]
    
    # Compare gradients
    all_close = True
    max_diff = 0
    
    print("Comparing gradients between the two approaches:")
    for i, (g1, g2) in enumerate(zip(grads_approach1, grads_approach2)):
        diff = torch.max(torch.abs(g1 - g2)).item()
        max_diff = max(max_diff, diff)
        is_close = torch.allclose(g1, g2, rtol=1e-7, atol=1e-7)
        all_close = all_close and is_close
        print(f"Layer {i}: {'✓' if is_close else '✗'} (max diff: {diff:.8f})")
    
    print(f"\nOverall result: {'✓ EQUIVALENT' if all_close else '✗ DIFFERENT'}")
    print(f"Maximum difference across all parameters: {max_diff:.8f}")
    
    return all_close

# Run the test
if __name__ == "__main__":
    result = test_wgan_gradient_equivalence()