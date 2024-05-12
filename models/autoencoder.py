import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.distributions.exponential import Exponential

## Define Snake Activation Function (From: https://arxiv.org/pdf/2006.08195)
# Code taken from  https://github.com/EdwardDixon/snake/blob/master/snake/activations.py
class Snake(nn.Module):
    '''         
    Implementation of the serpentine-like sine-based periodic activation function:
    .. math::
         Snake_a := x + \frac{1}{a} sin^2(ax) = x - \frac{1}{2a}cos{2ax} + \frac{1}{2a}
    This activation function is able to better extrapolate to previously unseen data,
    especially in the case of learning periodic functions

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
        
    Parameters:
        - a - trainable parameter
    
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
        
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, a=None, trainable=True):
        '''
        Initialization.
        Args:
            in_features: shape of the input
            a: trainable parameter
            trainable: sets `a` as a trainable parameter
            
            `a` is initialized to 1 by default, higher values = higher-frequency, 
            5-50 is a good starting point if you already think your data is periodic, 
            consider starting lower e.g. 0.5 if you think not, but don't worry, 
            `a` will be trained along with the rest of your model
        '''
        super(Snake,self).__init__()
        self.in_features = in_features 

        # Initialize `a`
        if a is not None:
            self.a = Parameter(torch.ones(self.in_features) * a) # create a tensor out of alpha
        else:            
            m = Exponential(torch.tensor([0.1]))
            self.a = Parameter((m.rsample(self.in_features)).squeeze()) # random init = mix of frequencies

        self.a.requiresGrad = trainable # set the training of `a` to true

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a* sin^2 (xa)
        '''
        return  x + (1.0/self.a) * torch.pow(torch.sin(x * self.a), 2)

class DownscaleResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_height, in_width):
        super(DownscaleResnetBlock, self).__init__()
        self.out_shape = (out_channels, in_height//2, in_width//2)
        self.activation_out = Snake(self.out_shape)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = self.downsample(x)
        out = self.activation_out(self.bn1(self.conv1(x)))
        out = self.activation_out(self.bn2(self.conv2(out)))
        out += identity
        return self.activation_out(self.bn3(out))
    
class UpscaleResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_height, in_width):
        super(UpscaleResnetBlock, self).__init__()
        in_shape = (in_channels, in_height, in_width)
        out_shape = (out_channels, in_height*2, in_width*2)
        self.activation = Snake(in_shape)
        self.activation_out = Snake(out_shape)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out += identity
        return self.activation_out(self.bn3(self.upsample(out)))

class VAE_Audio(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE_Audio, self).__init__()
        self.activation = nn.GELU()
        kernel_size = 16
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=kernel_size, stride=kernel_size//2, padding=kernel_size//4),
            self.activation,
            nn.Unflatten(1, (1, 128)),
            DownscaleResnetBlock(1, 2, 128, input_size//(kernel_size//2)),
            DownscaleResnetBlock(2, 4, 64, input_size//(2 * kernel_size//2)),
            DownscaleResnetBlock(4, 8, 32, input_size//(4 * kernel_size//2)),
            nn.Flatten(),
        )

        self.mu = nn.Linear(8*128//8*input_size//(8 * kernel_size//2), latent_size)
        self.logvar = nn.Linear( 8*128//8*input_size//(8 * kernel_size//2), latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 8*128//8*input_size//(8 * kernel_size//2)),
            self.activation,
            nn.Unflatten(1, (8, 16, input_size//(8 * kernel_size//2))),
            UpscaleResnetBlock(8, 4, 16, input_size//(8 * kernel_size//2)),
            UpscaleResnetBlock(4, 2, 32, input_size//(4 * kernel_size//2)),
            UpscaleResnetBlock(2, 1, 64, input_size//(2 * kernel_size//2)),
            nn.Flatten(1, 2),
            nn.ConvTranspose1d(128, 2, kernel_size=kernel_size, stride=kernel_size//2, padding=kernel_size//4),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar
    

## Define the loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD