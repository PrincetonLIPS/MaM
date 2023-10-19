import torch


class VarEstimator():
    def __init__(self, model):
        device = next(model.parameters()).device
        self.count = 0
        # iterate over all parameters in the model with gradient
        self.grad_bias = [torch.zeros_like(p.data, device=device) for p in model.parameters()]
        self.params = [torch.zeros_like(p.data, device=device) for p in model.parameters()]
        self.params2 = [torch.zeros_like(p.data, device=device) for p in model.parameters()]
        self.grad_true = [p.grad.clone() for p in model.parameters()]

    def update(self, model):
        x = [p.grad for p in model.parameters()]
        x2 = [p.grad**2 for p in model.parameters()]
        assert len(x) == len(self.params)
        for i in range(len(x)):
            self.grad_bias[i] = x[i] - self.grad_true[i] + self.grad_bias[i]
            self.params[i] = x[i] + self.params[i]
            self.params2[i] = x2[i] + self.params2[i]
        self.count += 1
    
    def get_variance(self):
        ret = []
        for i in range(len(self.params)-1):
            ret.append((self.params2[i]/self.count - (self.params[i]/self.count)**2).flatten())
        return torch.cat(ret).mean().item()
    
    def get_bias(self):
        ret = []
        for i in range(len(self.params)-1):
            ret.append((self.grad_bias[i]/self.count).flatten())
        return torch.cat(ret).mean().item()