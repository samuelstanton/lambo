import torch

from botorch.posteriors import Posterior

class EnsemblePosterior(Posterior):
    def __init__(self, X, model):
        super().__init__()
        self.X = X
        self.model = model

    @property
    def device(self):
        return list(self.model.parameters())[0].device

    @property
    def dtype(self):
        return list(self.model.parameters())[0].dtype

    @property
    def event_shape(self):
        return self.X.shape[:-1] + (1,)

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        if self.X.ndim == 1:
            y_outputs, _, _ = self.model(self.X, num_samples=sample_shape[0])
            return y_outputs
        elif self.X.ndim == 2:
            y_output_list = []
            for x in self.X:
                y_outputs, _, _ = self.model(x, num_samples=sample_shape[0])
                y_output_list.append(y_outputs)
            # TODO: fix the hard-coded permute here
            return torch.stack(y_output_list).permute(1, 0, 2, 3)
        
    @property
    def mean(self):
        return self.model(self.X)[1]
    
    @property
    def variance(self):
        return self.model(self.X)[2]
    
