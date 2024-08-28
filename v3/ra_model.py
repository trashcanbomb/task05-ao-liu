
import torch
import torch.nn as nn
import torch.nn.functional as F

class ADN(nn.Module):
    def __init__(self, channels, dropout_prob=0.0):
        super(ADN, self).__init__()
        self.norm = nn.InstanceNorm3d(channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.PReLU()

    def forward(self, x):
        return self.activation(self.dropout(self.norm(x)))

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_transpose=False, output_padding=0):
        super(Convolution, self).__init__()
        if is_transpose:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.adn = ADN(out_channels)

    def forward(self, x):
        return self.adn(self.conv(x))

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        super(ResidualUnit, self).__init__()
        self.conv1 = Convolution(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = Convolution(out_channels, out_channels, kernel_size, 1, padding)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_out, avg_out], dim=1))
        x = x * sa

        return x

class ReverseAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReverseAttention, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, y):
        y = F.interpolate(y, size=x.shape[2:], mode='trilinear', align_corners=True)
        y = self.conv(y)
        return x * ( - torch.sigmoid(y)) + x

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, channels=(64, 128, 256, 256, 512), num_classes=2):
        super(UNet3D, self).__init__()
        self.encoder1 = ResidualUnit(in_channels, channels[0], stride=2)
        self.encoder2 = ResidualUnit(channels[0], channels[1], stride=2)
        self.encoder3 = ResidualUnit(channels[1], channels[2], stride=2)
        self.encoder4 = ResidualUnit(channels[2], channels[3], stride=2)
        self.bottleneck = ResidualUnit(channels[3], channels[4], stride=1)

        self.cbam1 = CBAM(channels[0])
        self.cbam2 = CBAM(channels[1])
        self.cbam3 = CBAM(channels[2])
        self.cbam4 = CBAM(channels[3])
        self.cbam_bottleneck = CBAM(channels[4])

        self.upconv1 = Convolution(channels[4] + channels[3], channels[3], 3, 2, 1, is_transpose=True, output_padding=1)
        self.decoder1 = ResidualUnit(channels[3], channels[3], stride=1)

        self.upconv2 = Convolution(channels[3] + channels[2], channels[2], 3, 2, 1, is_transpose=True, output_padding=1)
        self.decoder2 = ResidualUnit(channels[2], channels[2], stride=1)

        self.upconv3 = Convolution(channels[2] + channels[1], channels[1], 3, 2, 1, is_transpose=True, output_padding=1)
        self.decoder3 = ResidualUnit(channels[1], channels[1], stride=1)

        self.upconv4 = Convolution(channels[1] + channels[0], num_classes, 3, 2, 1, is_transpose=True, output_padding=1)
        self.decoder4 = ResidualUnit(num_classes, num_classes, stride=1)

        self.reverse_attention2 = ReverseAttention(channels[2], 256)
        self.decoder_reverse_attention2 = ResidualUnit(channels[2], num_classes, stride=1)
        self.reverse_attention3 = ReverseAttention(channels[2], 128)
        self.decoder_reverse_attention3 = ResidualUnit(channels[1], num_classes, stride=1)
        self.reverse_attention4 = ReverseAttention(channels[1], num_classes)
        self.decoder_reverse_attention4 = ResidualUnit(channels[0], num_classes, stride=1)

    def forward(self, x):
        e1 = self.cbam1(self.encoder1(x))
        e2 = self.cbam2(self.encoder2(e1))
        e3 = self.cbam3(self.encoder3(e2))
        e4 = self.cbam4(self.encoder4(e3))
        b = self.cbam_bottleneck(self.bottleneck(e4))

        d1 = self.decoder1(self.upconv1(torch.cat([b, e4], dim=1)))
        d1_upsampled = F.interpolate(d1, size=e3.shape[2:], mode='trilinear', align_corners=True)

        d2 = self.decoder2(self.upconv2(torch.cat([d1_upsampled, e3], dim=1)))
        ra2 = self.reverse_attention2(d2, d1_upsampled)

        d2_upsampled = F.interpolate(ra2, size=e2.shape[2:], mode='trilinear', align_corners=True)
        d3 = self.decoder3(self.upconv3(torch.cat([d2_upsampled, e2], dim=1)))
        ra3 = self.reverse_attention3(d3, d2_upsampled)

        d3_upsampled = F.interpolate(ra3, size=e1.shape[2:], mode='trilinear', align_corners=True)
        d4 = self.decoder4(self.upconv4(torch.cat([d3_upsampled, e1], dim=1)))
        ra4 = self.reverse_attention4(d4, d3_upsampled)

        ra2 = self.decoder_reverse_attention2(ra2)
        ra2 = F.interpolate(ra2, size=d4.shape[2:], mode='trilinear', align_corners=True)
        ra3 = self.decoder_reverse_attention3(ra3)
        ra3 = F.interpolate(ra3, size=d4.shape[2:], mode='trilinear', align_corners=True)


        return d4, ra2, ra3, ra4

# # Example usage
# model = UNet3D(in_channels=2, channels=(64, 128, 256, 256, 512), num_classes=2)
# x = torch.randn(1, 2, 64, 64, 64)  # Example input
# output, ra2_output, ra3_output, ra4_output = model(x)
# print(output.shape, ra2_output.shape, ra3_output.shape, ra4_output.shape)