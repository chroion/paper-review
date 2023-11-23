import torch
import torch.nn as nn
import torch.nn.functional as F
from rfft2d import Rfft2d

EPS = 1e-10

def softmax(a, b, factor=1):
    concat = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
    softmax_factors = F.softmax(concat * factor, dim=-1)
    return a * softmax_factors[:,:,:,:,0] + b * softmax_factors[:,:,:,:,1]

class WatsonDistanceFft(nn.Module):
    """
    Loss function based on Watsons perceptual distance.
    Based on FFT quantization
    """
    def __init__(self, blocksize=8, trainable=False, reduction='sum'):
        """
        Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform 
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
        """
        super().__init__()
        self.trainable = trainable
        
        # input mapping
        blocksize = torch.as_tensor(blocksize)
        
        # module to perform 2D blockwise rFFT
        self.add_module('fft', Rfft2d(blocksize=blocksize.item(), interleaving=False))
    
        # parameters
        self.weight_size = (blocksize, blocksize // 2 + 1)
        self.blocksize = nn.Parameter(blocksize, requires_grad=False)
        # init with uniform QM
        self.t_tild = nn.Parameter(torch.zeros(self.weight_size), requires_grad=trainable)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=trainable) # luminance masking
        w = torch.tensor(0.2) # contrast masking
        self.w_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable) # inverse of sigmoid
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=trainable) # pooling
        
        # phase weights
        self.w_phase_tild = nn.Parameter(torch.zeros(self.weight_size) -2., requires_grad=trainable)
        
        # dropout for training
        self.dropout = nn.Dropout(0.5 if trainable else 0)
        
        # reduction
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception('Reduction "{}" not supported. Valid values are: "sum", "none".'.format(reduction))

    @property
    def t(self):
        # returns QM
        qm = torch.exp(self.t_tild)
        return qm
    
    @property
    def w(self):
        # return luminance masking parameter
        return torch.sigmoid(self.w_tild)
    
    @property
    def w_phase(self):
        # return weights for phase
        w_phase =  torch.exp(self.w_phase_tild)
        # set weights of non-phases to 0
        if not self.trainable:
            w_phase[0,0] = 0.
            w_phase[0,self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1,self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1, 0] = 0.
        return w_phase
    
    def forward(self, input):
        # fft
        c1 = self.fft(input)
        
        return c1
    
