import torch
from torch import nn
from torch.nn import functional as F

class AttentionBlock(nn.Module):
    """Attention block for U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        
        self.bn=None
        if norm:
            self.bn = nn.BatchNorm2d(outplanes)
        
    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)
        
        if self.bn is not None:
            fx = self.bn(fx)
            
        return fx

    
class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)       
        
        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
            
    def forward(self, x):
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)
            
        return fx

class AttentionUnetGenerator(nn.Module):
    """Unet-like Encoder-Decoder model with Attention Blocks"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.encoder1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        
        # Decoder
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder6 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder5 = DecoderBlock(2*512, 512)
        self.decoder4 = DecoderBlock(2*512, 256)
        self.decoder3 = DecoderBlock(2*256, 128)
        self.decoder2 = DecoderBlock(2*128, 64)
        self.decoder1 = nn.ConvTranspose2d(2*64, out_channels, kernel_size=4, stride=2, padding=1)
        
        # Attention blocks
        self.att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)  # Matches e6 and d7
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)  # Matches e5 and d6
        self.att3 = AttentionBlock(F_g=512, F_l=512, F_int=256)  # Matches e4 and d5
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)  # Matches e3 and d4
        
    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        
        # Decoder path
        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        
        d7 = self.decoder7(d8)
        e6 = self.att5(g=d7, x=e6)  # Apply attention before concatenation
        d7 = torch.cat([d7, e6], dim=1)
        
        d6 = self.decoder6(d7)
        e5 = self.att4(g=d6, x=e5)  # Apply attention before concatenation
        d6 = torch.cat([d6, e5], dim=1)
        
        d5 = self.decoder5(d6)
        e4 = self.att3(g=d5, x=e4)  # Apply attention before concatenation
        d5 = torch.cat([d5, e4], dim=1)
        
        d4 = self.decoder4(d5)
        e3 = self.att2(g=d4, x=e3)  # Apply attention before concatenation
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)

