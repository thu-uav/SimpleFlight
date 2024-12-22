import torch

class Lemniscate():
    def __init__(self, T: float = 15.0, origin: torch.Tensor = torch.zeros(3), device: str = 'cpu'):
        self.T = T
        self.device = device
        self.origin = origin
    
    def reset(self, idx: torch.Tensor = None):
        pass
        
    def pos(self, t):
        sin_t = torch.sin(2 * torch.pi * t / self.T)
        cos_t = torch.cos(2 * torch.pi * t / self.T)

        x = torch.stack([
            cos_t, sin_t * cos_t, torch.zeros_like(t)
        ], dim=-1)
        
        # v = torch.stack([
        #     -2 * torch.pi / T * sin_t, 2 * torch.pi / T * torch.cos(4 * torch.pi * t / T), torch.zeros_like(t)
        # ], dim=-1)
        
        return (x + self.origin).to(self.device)