import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    UNet for Semantic Segmentation.

    Channel schedule (base_filters=64):
        enc1  64  |  enc2 128  |  enc3 256  |  enc4 512
        bottleneck 1024
        dec4 512  |  dec3 256  |  dec2 128  |  dec1 64

    Args:
        in_channels  (int)  : Input image channels (e.g. 1 or 3).
        num_classes  (int)  : Number of segmentation classes.
        base_filters (int)  : Channels in the first encoder block (doubles each level).
        bilinear     (bool) : True -> bilinear upsample + 1x1 conv.
                              False -> learnable ConvTranspose2d.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_filters: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = self._double_conv(in_channels, f)
        self.enc2 = self._double_conv(f,     f * 2)
        self.enc3 = self._double_conv(f * 2, f * 4)
        self.enc4 = self._double_conv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck - always outputs f*16 channels
        self.bottleneck = self._double_conv(f * 8, f * 16)

        # Decoder
        # _UpBlock(in_ch, out_ch): upsample->in_ch//2, cat skip (in_ch//2), double_conv->out_ch
        self.up4 = _UpBlock(f * 16, f * 8,  bilinear)
        self.up3 = _UpBlock(f * 8,  f * 4,  bilinear)
        self.up2 = _UpBlock(f * 4,  f * 2,  bilinear)
        self.up1 = _UpBlock(f * 2,  f,      bilinear)

        # Head
        self.head = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.head(d1)

    @staticmethod
    def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            ## Deactivate double conv
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
        )


class _UpBlock(nn.Module):
    """
    Decoder block:
      1. Upsample x2  ->  in_ch // 2  channels
      2. Cat skip     ->  in_ch       channels  (in_ch//2 up + in_ch//2 skip)
      3. Double conv  ->  out_ch      channels
    """
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool):
        super().__init__()
        half = in_ch // 2
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, half, kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, half, kernel_size=2, stride=2)
        self.conv = UNet._double_conv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = UNet(in_channels=3, num_classes=10, base_filters=64, bilinear=True).to(device)
    print(model)
    dummy  = torch.randn(2, 3, 256, 256, device=device)
    logits = model(dummy)
    n      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(logits.shape)}")
    print(f"Params : {n:,}")
    assert logits.shape == (2, 10, 256, 256)
    print("Forward pass OK")