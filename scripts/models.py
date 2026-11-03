"""Segmentation models. Every model takes [B, 3, 128, 800] and returns 4 logit maps [B, 4, 128, 800]."""
import torch
import torch.nn as nn

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet3(nn.Module):
    """3 stage U-Net built from scratch (encoder, bottleneck, decoder with skip connections)."""

    def __init__(self, b=16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(3, b), block(b, 2 * b), block(2 * b, 4 * b)
        self.pool = nn.MaxPool2d(2)
        self.mid = block(4 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), block(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), block(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), block(2 * b, b)
        self.out = nn.Conv2d(b, 4, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        x = self.d3(torch.cat([self.u3(m), e3], 1))
        x = self.d2(torch.cat([self.u2(x), e2], 1))
        x = self.d1(torch.cat([self.u1(x), e1], 1))
        return self.out(x)


class Normalized(nn.Module):
    """Applies ImageNet normalisation before the network."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build_model(name, pretrained=True):
    """scratch_unet: built from scratch | resnet34_unet: pretrained | effb2_fpn: pretrained, my third model
    effb3_fpn: bigger EfficientNet used in milestone 4"""
    if name == "scratch_unet":
        return Normalized(UNet3(b=32))
    if name == "baseline_unet":
        return UNet3(b=16)
    import segmentation_models_pytorch as smp
    weights = "imagenet" if pretrained else None
    if name == "resnet34_unet":
        return Normalized(smp.Unet("resnet34", encoder_weights=weights, in_channels=3, classes=4))
    if name == "effb2_fpn":
        return Normalized(smp.FPN("efficientnet-b2", encoder_weights=weights, in_channels=3, classes=4))
    if name == "effb3_fpn":
        return Normalized(smp.FPN("efficientnet-b3", encoder_weights=weights, in_channels=3, classes=4))
    raise ValueError(f"unknown model {name}")
