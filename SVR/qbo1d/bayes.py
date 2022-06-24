import torch

from sklearn.preprocessing import StandardScaler

def _make_torch_scaler(ScikitClass):   
    class TorchScaler(ScikitClass):
        def transform(self, data):
            return torch.tensor(super().transform(data)).float()
        
        def inverse_transform(self, data):
            return torch.tensor(super().inverse_transform(data)).float()
        
    return TorchScaler

TorchStandardScaler = _make_torch_scaler(StandardScaler)