import torch
from torch import nn

class EncoderBlock(nn.Module):
    """Encoder block with attention capability"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes) if norm else None

    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)
        if self.bn is not None:
            fx = self.bn(fx)
        return fx

class DecoderBlock(nn.Module):
    """Decoder block with attention capability"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True) if dropout else None

    def forward(self, x):
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)
        if self.dropout:
            fx = self.dropout(fx)
        return fx

class AttentionBlock(nn.Module):
    """Attention block for Attention U-Net"""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi * x

class AttentionUNetGenerator(nn.Module):
    """Attention U-Net Generator"""
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

        # Decoder with attention
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.attention7 = AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.decoder7 = DecoderBlock(2 * 512, 512, dropout=True)
        self.attention6 = AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.decoder6 = DecoderBlock(2 * 512, 512, dropout=True)
        self.attention5 = AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.decoder5 = DecoderBlock(2 * 512, 512)
        self.attention4 = AttentionBlock(f_g=512, f_l=256, f_int=128)
        self.decoder4 = DecoderBlock(2 * 256, 256)
        self.attention3 = AttentionBlock(f_g=256, f_l=128, f_int=64)
        self.decoder3 = DecoderBlock(2 * 128, 128)
        self.attention2 = AttentionBlock(f_g=128, f_l=64, f_int=32)
        self.decoder2 = DecoderBlock(2 * 64, 64)
        self.decoder1 = nn.ConvTranspose2d(2 * 64, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoder forward
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        # Decoder forward with attention
        d8 = self.decoder8(e8)
        e7 = self.attention7(d8, e7)
        d8 = torch.cat([d8, e7], dim=1)

        d7 = self.decoder7(d8)
        e6 = self.attention6(d7, e6)
        d7 = torch.cat([d7, e6], dim=1)

        d6 = self.decoder6(d7)
        e5 = self.attention5(d6, e5)
        d6 = torch.cat([d6, e5], dim=1)

        d5 = self.decoder5(d6)
        e4 = self.attention4(d5, e4)
        d5 = torch.cat([d5, e4], dim=1)

        d4 = self.decoder4(d5)
        e3 = self.attention3(d4, e3)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.decoder3(d4)
        e2 = self.attention2(d3, e2)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.decoder1(d2)

        return torch.tanh(d1)

