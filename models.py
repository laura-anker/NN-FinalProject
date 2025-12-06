import torch
import torch.nn as nn

# Encoder block that downsamples by a factor of 2
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not batchnorm)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
# Decoder block that upsamples by a factor of 2
class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 5-layer U-Net generator
# Transforms the input L channel into ab chrominance by encoding it through successive 
# downsampling to capture global structure, then decoding back to full resolution.
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=64):
        super().__init__()

        # Encoder: progressively downsamples (224 → 7) and increases the number of channels to learn hierarchical features
        self.down1 = DownBlock(in_channels, features, batchnorm=False)
        self.down2 = DownBlock(features, features * 2)
        self.down3 = DownBlock(features * 2, features * 4)
        self.down4 = DownBlock(features * 4, features * 8)
        self.down5 = DownBlock(features * 8, features * 8)

        # Bottleneck: maintains the 7×7 spatial size, further mixing high-level features across channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder: progressively upsamples (7 → 224) and fuses encoder skips to recover spatial detail
        # Note: in_c values account for concatenation with the corresponding encoder feature maps
        self.up5 = UpBlock(features * 8, features * 8, dropout=True)
        self.up4 = UpBlock(features * 16, features * 4, dropout=True)
        self.up3 = UpBlock(features * 8, features * 2)
        self.up2 = UpBlock(features * 4, features)
        self.up1 = UpBlock(features * 2, features)

        # Final: maps top-level decoder features to the desired output channels
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

        # Activation: projects the output to [-1, 1] for normalized ab channels
        self.activation = nn.Tanh()

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)               # 64 @ 112x112
        d2 = self.down2(d1)              # 128 @ 56x56
        d3 = self.down3(d2)              # 256 @ 28x28
        d4 = self.down4(d3)              # 512 @ 14x14
        d5 = self.down5(d4)              # 512 @ 7x7 

        # Bottleneck
        b = self.bottleneck(d5)          # 512 @ 7x7

        # Decoder and skips
        u5 = self.up5(b)                 # 512 @ 14x14
        u5 = torch.cat([u5, d4], dim=1)  # 1024 @ 14x14

        u4 = self.up4(u5)                # 512 @ 28x28
        u4 = torch.cat([u4, d3], dim=1)  # 1024 @ 28x28

        u3 = self.up3(u4)                # 256 @ 56x56
        u3 = torch.cat([u3, d2], dim=1)  # 512 @ 56x56

        u2 = self.up2(u3)                # 128 @ 112x112
        u2 = torch.cat([u2, d1], dim=1)  # 256 @ 112x112

        u1 = self.up1(u2)                # 64 @ 224x224

        # Final and activation
        out = self.final(u1)             # out_channels @ 224x224
        out = self.activation(out)
        return out

# Patch-based discriminator
# Evaluates local patches, producing a spatial map of real/fake scores to enforce 
# locally realistic color and texture in the generator’s output.
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.net = nn.Sequential(
            # First downsample (224 → 112) to begin building patch receptive fields
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Second downsample (112 → 56) to grow the receptive field
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),

            # Third downsample (56 → 28) to further grow the receptive field
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2),

            # Convolution to increase the number of channels thus refining high-level patch features (no further downsampling)
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2),

            # Final convolution to produce a 1-channel map where each entry is a patch-realism score
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)
