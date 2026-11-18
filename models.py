import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with optional normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, 
                 padding=1, normalize=True, activation=True, negative_slope=0.2):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=not normalize)
        self.normalize = normalize
        self.activation = activation
        
        if normalize:
            self.norm = nn.BatchNorm2d(out_channels)
        
        if activation:
            self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        if self.activation:
            x = self.leaky_relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for generator decoder"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    """U-Net generator for Pix2Pix"""
    def __init__(self, in_channels=3, out_channels=3, num_filters=64, num_blocks=7):
        super().__init__()
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        
        # Encoder: downsample with explicit channel tracking
        self.encoder_channels = []
        self.encoder = nn.ModuleList()
        channels = in_channels
        for i in range(8):  # 8 downsampling layers
            out_ch = min(num_filters * (2 ** i), 512)
            self.encoder_channels.append(out_ch)
            self.encoder.append(ConvBlock(
                channels, out_ch, 
                normalize=(i > 0),  # First layer no batch norm
                activation=True
            ))
            channels = out_ch
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        
        # Decoder: upsample with skip connections
        self.decoder = nn.ModuleList()
        # Build decoder from deepest to shallowest
        decoder_channels = []
        for i in range(8):
            # Skip connection: concatenate with encoder output from corresponding layer
            encoder_out_ch = self.encoder_channels[7 - i]  # Matching encoder layer
            skip_ch = encoder_out_ch
            
            # Input to this decoder layer: previous output (or residual output) + skip connection
            if i == 0:
                in_ch = channels  # Output from residual blocks
            else:
                in_ch = decoder_channels[-1]  # Output from previous decoder layer
            
            concatenated_ch = in_ch + skip_ch  # After concatenation
            
            # Output channel calculation
            if i < 7:
                out_ch = in_ch // 2  # Halve channels going up
            else:
                out_ch = out_channels
            
            decoder_channels.append(out_ch)
            
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(concatenated_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch) if i < 7 else nn.Identity(),
                nn.ReLU(inplace=True) if i < 7 else nn.Tanh()
            ))
    
    def forward(self, x):
        # Encoder with skip connections
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
        
        # Residual blocks
        x = self.residual_blocks(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            # Concatenate with corresponding encoder output
            skip_connection = encoder_outputs[7 - i]
            x = torch.cat([x, skip_connection], dim=1)
            x = layer(x)
        
        return x


class Discriminator(nn.Module):
    """PatchGAN discriminator"""
    def __init__(self, in_channels=6, num_filters=64, num_layers=3):
        super().__init__()
        
        layers = []
        channels = in_channels
        
        # Initial layer
        layers.append(ConvBlock(
            channels, num_filters,
            normalize=False,  # No batch norm on first layer
            activation=True
        ))
        channels = num_filters
        
        # Intermediate layers
        for i in range(num_layers - 1):
            out_channels = min(channels * 2, 512)
            layers.append(ConvBlock(
                channels, out_channels,
                normalize=True,
                activation=True
            ))
            channels = out_channels
        
        # Output layer
        layers.append(nn.Conv2d(channels, 1, 4, 1, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Pix2PixModel(nn.Module):
    """Complete Pix2Pix model"""
    def __init__(self, config):
        super().__init__()
        
        self.generator = Generator(
            in_channels=config['model']['generator']['input_channels'],
            out_channels=config['model']['generator']['output_channels'],
            num_filters=config['model']['generator']['num_filters'],
            num_blocks=config['model']['generator']['num_blocks']
        )
        
        self.discriminator = Discriminator(
            in_channels=config['model']['discriminator']['input_channels'],
            num_filters=config['model']['discriminator']['num_filters'],
            num_layers=config['model']['discriminator']['num_layers']
        )
    
    def forward(self, A):
        """Generate B from A"""
        return self.generator(A)
