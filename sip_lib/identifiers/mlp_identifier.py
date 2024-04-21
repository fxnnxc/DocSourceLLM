import torch 
import torch.nn as nn 
from omegaconf import OmegaConf
from sip_lib.identifiers.utils import layer_init

class MLPIdentifier(nn.Module):
    def __init__(self, hidden_size, linear_hidden_size, linear_activation, num_outputs, **kwargs):
        """_summary_
        Args:
            hidden_size (_type_): gpt output dimension size
            linear_hidden_size (_type_): dimension of linear layer
            num_binaries (_type_): _description_
            linear_activation (_type_): _description_
        """
        self.num_outputs = num_outputs
            
        self.hidden_size = hidden_size
        self.config = OmegaConf.create({
            "linear_hidden_size": linear_hidden_size,
            "num_outputs" : self.num_outputs,
            'linear_activation' : linear_activation,
        })
        if self.config.linear_activation == "relu":
            activation_fn = nn.ReLU
        elif self.config.linear_activation =="gelu":
            activation_fn = nn.GELU
        elif self.config.linear_activation =="tanh":
            activation_fn = nn.Tanh
            
        super().__init__()
        self.config.linear_hidden_size = eval(self.config.linear_hidden_size)
        net = []
        in_features = self.hidden_size
        for i in range(len(self.config.linear_hidden_size)):
            hidden = int(self.config.linear_hidden_size[i])
            net.append(layer_init(nn.Linear(in_features, hidden)))
            net.append(activation_fn())
            in_features = hidden
        net.append(layer_init(nn.Linear(in_features, self.num_outputs)))
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        raise NotImplementedError(0)
        
class MLPIdentifier(MLPIdentifier):
    def __init__(self, hidden_size, linear_hidden_size, linear_activation, num_outputs, **kwargs):
        super().__init__(hidden_size, linear_hidden_size, linear_activation, num_outputs, **kwargs)
        
    def forward(self, x):
        return self.net(x)