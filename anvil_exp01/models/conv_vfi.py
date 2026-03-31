"""Model definitions for ANVIL VFI experiments.

Provides architectures and a model registry for exp2 (route comparison),
exp3 (capacity sweep), Phase 3 ablations (resblock / confidence gate),
and Phase 4 ceiling test (NAFNet-based strong refinement).

Most models output a 3-channel residual (final = blend + residual).
Models with ``returns_frame = True`` output the final frame directly
(they compute their own blend internally).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVFI_Small(nn.Module):
    """8-layer plain conv stack, 24ch hidden.

    Architecture: entry(in_ch→24) + 6×mid(24→24) + exit(24→3).
    """

    def __init__(self, in_ch: int = 8):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, 24, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(6):
            layers.append(nn.Conv2d(24, 24, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(24, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)
        # Zero-init last conv so initial residual = 0 (prediction starts at blend)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvVFI_Large(nn.Module):
    """12-layer dilated conv stack, 48ch hidden.

    Architecture: entry(in_ch→48) + 10×dilated(48→48) + exit(48→3).
    Dilation pattern [1,1,2,2,4,4,8,8,4,2] expands receptive field
    without increasing parameter count per layer.
    """

    def __init__(self, in_ch: int = 8):
        super().__init__()
        ch = 48
        dilations = [1, 1, 2, 2, 4, 4, 8, 8, 4, 2]
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for d in dilations:
            layers.append(nn.Conv2d(ch, ch, 3, padding=d, dilation=d))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(ch, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvVFI_Plain(nn.Module):
    """Configurable plain conv stack for capacity sweep.

    Architecture: entry(in_ch→ch) + (n_layers-2)×mid(ch→ch) + exit(ch→3).
    """

    def __init__(self, in_ch: int = 8, channels: int = 16, n_layers: int = 6):
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvVFI_UNet(nn.Module):
    """3-level U-Net with stride-2 conv down / transposed conv up.

    Architecture: 3 encoder levels (each 2×conv+ReLU), stride-2 downsampling,
    symmetric decoder with skip connections, 1×1 output conv.
    Channel progression: base_ch → 2×base_ch → 4×base_ch.
    """

    def __init__(self, in_ch: int = 8, base_ch: int = 16):
        super().__init__()
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch1, ch1, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(ch1, ch1, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch2, ch2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ch2, ch2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch2, ch3, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch3, ch3, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(ch3, ch2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(ch2 * 2, ch2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch2, ch2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(ch2, ch1, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch1 * 2, ch1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch1, ch1, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(ch1, 3, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class DWBlock(nn.Module):
    """DWConv + 1×1 expansion block with optional channel attention.

    DWConv 3×3 → ReLU → 1×1 expand → ReLU → 1×1 compress → residual skip.
    Optionally adds SCA (Simple Channel Attention: GAP → 1×1 → scale).
    All ops are HTP-native (Conv + ReLU, no LN/GELU/Resize).
    """

    def __init__(self, ch: int, expand: int = 4, use_sca: bool = True):
        super().__init__()
        mid = ch * expand
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw1 = nn.Conv2d(ch, mid, 1)
        self.pw2 = nn.Conv2d(mid, ch, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1),
        ) if use_sca else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.dw(x))
        y = self.relu(self.pw1(y))
        y = self.pw2(y)
        if self.sca is not None:
            y = y * self.sca(y)
        return x + y


class ConvVFI_UNetV2(nn.Module):
    """HTP-optimized 4-level U-Net with DWBlock bottleneck.

    Key improvements over ConvVFI_UNet:
    - 4 levels (vs 3): deeper downsampling for larger receptive field
    - DWConv + 1×1 pointwise blocks: more parameter-efficient than 3×3 conv
    - Deep bottleneck: many blocks at lowest resolution (cheap on HTP)
    - Channel attention (SCA): global context with minimal cost
    - All ops HTP-native: Conv + ReLU only, no LN/GELU/Resize/grid_sample

    Channel progression: base_ch → 2× → 4× → 8×.
    """

    def __init__(
        self,
        in_ch: int = 6,
        base_ch: int = 24,
        bottleneck_blocks: int = 8,
        enc_blocks: tuple[int, ...] = (1, 1, 1, 1),
        dec_blocks: tuple[int, ...] = (1, 1, 1, 1),
        expand: int = 4,
        use_sca: bool = True,
    ):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(4)]  # e.g. 24, 48, 96, 192

        # Intro: project input to base_ch
        self.intro = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder: DWBlocks + stride-2 down
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(4):
            self.encoders.append(
                nn.Sequential(*[DWBlock(chs[i], expand, use_sca) for _ in range(enc_blocks[i])])
            )
            if i < 3:
                self.downs.append(nn.Sequential(
                    nn.Conv2d(chs[i], chs[i + 1], 2, stride=2),
                    nn.ReLU(inplace=True),
                ))
            else:
                # Last encoder, down into bottleneck (same channels)
                self.downs.append(nn.Sequential(
                    nn.Conv2d(chs[i], chs[i], 2, stride=2),
                    nn.ReLU(inplace=True),
                ))

        # Bottleneck: deep stack at 1/16 resolution
        self.bottleneck = nn.Sequential(
            *[DWBlock(chs[3], expand, use_sca) for _ in range(bottleneck_blocks)]
        )

        # Decoder: upsample + skip concat + DWBlocks
        self.ups = nn.ModuleList()
        self.dec_proj = nn.ModuleList()  # project concat(up + skip) back to ch
        self.decoders = nn.ModuleList()
        for i in range(3, -1, -1):
            up_in = chs[3] if i == 3 else chs[i + 1]
            self.ups.append(nn.ConvTranspose2d(up_in, chs[i], 2, stride=2))
            # After concat with skip: 2 × chs[i] → chs[i]
            self.dec_proj.append(nn.Sequential(
                nn.Conv2d(chs[i] * 2, chs[i], 1),
                nn.ReLU(inplace=True),
            ))
            self.decoders.append(
                nn.Sequential(*[DWBlock(chs[i], expand, use_sca) for _ in range(dec_blocks[3 - i])])
            )

        # Output: zero-init for residual learning
        self.out_conv = nn.Conv2d(chs[0], 3, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # Pad to multiple of 2^num_levels (4 levels → 16)
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.intro(x)

        # Encoder
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, proj, dec, skip in zip(
            self.ups, self.dec_proj, self.decoders, reversed(skips)
        ):
            x = up(x)
            x = proj(torch.cat([x, skip], dim=1))
            x = dec(x)

        x = self.out_conv(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


class ResBlock(nn.Module):
    """Residual block: x + F(x). 1x1 projection when channels differ.

    Args:
        norm: "none" | "bn". BN fuses into Conv at deployment (zero cost).

    ``fuse_for_deploy()`` folds BN parameters into the preceding Conv
    weights/biases, eliminating BN from the inference graph.  The residual
    Add is **not** eliminated — RepVGG-style identity-skip merge is invalid
    for two-layer blocks because conv2 receives relu(conv1(x)), not x.
    """

    def __init__(self, in_ch: int, out_ch: int, norm: str = "none"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch) if norm == "bn" else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch) if norm == "bn" else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + identity)

    def fuse_for_deploy(self) -> None:
        """Fuse BN into Conv for zero-cost inference.

        Only BN fusion is applied; the skip Add remains in the graph.
        """
        for bn_name, conv_name in [("bn1", "conv1"), ("bn2", "conv2")]:
            bn = getattr(self, bn_name)
            if isinstance(bn, nn.BatchNorm2d):
                conv = getattr(self, conv_name)
                _fuse_conv_bn_inplace(conv, bn)
                setattr(self, bn_name, nn.Identity())


class ConvVFI_UNetV3(nn.Module):
    """INT8-native 4-level U-Net with standard Conv 3x3 + ResBlocks.

    Designed from INT8 HTP per-op profiling (Phase 5b):
    - Standard Conv 3x3 only (compute-bound, 4.0x INT8 speedup)
    - No DWConv/SCA/attention (memory-bound, poor INT8 acceleration)
    - ReLU only (hardware-native zero overhead on HTP)
    - 4 levels for large receptive field at 1/16 resolution
    - Deep bottleneck: many ResBlocks at lowest resolution (nearly free)
    - ResBlock skip connections for better gradient flow (+0.29 dB proven)

    Channel progression: base_ch * ch_mults, default (1,2,4,4) caps at
    4x to keep deep levels affordable while adding capacity through
    more blocks at low resolution.
    """

    def __init__(
        self,
        in_ch: int = 6,
        base_ch: int = 24,
        ch_mults: tuple[int, ...] = (1, 2, 4, 4),
        enc_blocks: tuple[int, ...] = (1, 1, 1, 2),
        dec_blocks: tuple[int, ...] = (1, 1, 1, 1),
        bottleneck_blocks: int = 4,
    ):
        super().__init__()
        num_levels = len(ch_mults)
        self.pad_multiple = 2 ** num_levels
        chs = [base_ch * m for m in ch_mults]

        # Intro: project input to base channels
        self.intro = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder: ResBlocks + stride-2 down
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(num_levels):
            self.encoders.append(
                nn.Sequential(*[ResBlock(chs[i], chs[i]) for _ in range(enc_blocks[i])])
            )
            next_ch = chs[i + 1] if i < num_levels - 1 else chs[i]
            self.downs.append(nn.Sequential(
                nn.Conv2d(chs[i], next_ch, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ))

        # Bottleneck at 1/(2^num_levels) resolution
        self.bottleneck = nn.Sequential(
            *[ResBlock(chs[-1], chs[-1]) for _ in range(bottleneck_blocks)]
        )

        # Decoder: upsample + skip concat + project + ResBlocks
        self.ups = nn.ModuleList()
        self.dec_proj = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            up_in = chs[-1] if i == num_levels - 1 else chs[i + 1]
            self.ups.append(nn.ConvTranspose2d(up_in, chs[i], 2, stride=2))
            self.dec_proj.append(nn.Sequential(
                nn.Conv2d(chs[i] * 2, chs[i], 1),
                nn.ReLU(inplace=True),
            ))
            self.decoders.append(
                nn.Sequential(*[ResBlock(chs[i], chs[i])
                                for _ in range(dec_blocks[num_levels - 1 - i])])
            )

        # Output: zero-init for residual learning
        self.out_conv = nn.Conv2d(chs[0], 3, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # Pad to multiple of 2^num_levels
        pad_h = (self.pad_multiple - H % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - W % self.pad_multiple) % self.pad_multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.intro(x)

        # Encoder
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, proj, dec, skip in zip(
            self.ups, self.dec_proj, self.decoders, reversed(skips)
        ):
            x = up(x)
            x = proj(torch.cat([x, skip], dim=1))
            x = dec(x)

        x = self.out_conv(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


class ConvVFI_UNetV3b(nn.Module):
    """V3 iteration: profiling-driven optimizations over ConvVFI_UNetV3.

    Changes from v3:
    - U-Net skip: concat+proj → element-wise add (saves Concat + Conv1x1)
    - Optional BN in ResBlock (fuses into Conv at deploy, zero cost)
    - No reflect padding (caller provides padded input at deploy)

    Same core: standard Conv 3x3 + ResBlock, 4-level U-Net, deep bottleneck.

    Call ``fuse_for_deploy()`` before ONNX export to fold BN into Conv
    (zero-cost normalization).  The residual Add ops remain in the graph.
    """

    def __init__(
        self,
        in_ch: int = 6,
        base_ch: int = 24,
        ch_mults: tuple[int, ...] = (1, 2, 4, 4),
        enc_blocks: tuple[int, ...] = (1, 1, 1, 2),
        dec_blocks: tuple[int, ...] = (1, 1, 1, 1),
        bottleneck_blocks: int = 4,
        norm: str = "none",
    ):
        super().__init__()
        num_levels = len(ch_mults)
        self.pad_multiple = 2 ** num_levels
        chs = [base_ch * m for m in ch_mults]

        # Intro
        self.intro = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(num_levels):
            self.encoders.append(
                nn.Sequential(*[ResBlock(chs[i], chs[i], norm=norm)
                                for _ in range(enc_blocks[i])])
            )
            next_ch = chs[i + 1] if i < num_levels - 1 else chs[i]
            self.downs.append(nn.Sequential(
                nn.Conv2d(chs[i], next_ch, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResBlock(chs[-1], chs[-1], norm=norm)
              for _ in range(bottleneck_blocks)]
        )

        # Decoder: upsample + additive skip (no concat/proj)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            up_in = chs[-1] if i == num_levels - 1 else chs[i + 1]
            self.ups.append(nn.ConvTranspose2d(up_in, chs[i], 2, stride=2))
            self.decoders.append(
                nn.Sequential(*[ResBlock(chs[i], chs[i], norm=norm)
                                for _ in range(dec_blocks[num_levels - 1 - i])])
            )

        # Output
        self.out_conv = nn.Conv2d(chs[0], 3, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def fuse_for_deploy(self) -> int:
        """Fuse BN into Conv on all ResBlocks.

        Returns number of fused blocks.
        """
        count = 0
        for m in self.modules():
            if isinstance(m, ResBlock):
                m.fuse_for_deploy()
                count += 1
        return count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        pad_h = (self.pad_multiple - H % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - W % self.pad_multiple) % self.pad_multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.intro(x)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x) + skip  # additive skip, no concat
            x = dec(x)

        x = self.out_conv(x)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


def _fuse_conv_bn_inplace(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """Fuse Conv→BN into a single Conv (standard deployment pattern)."""
    with torch.no_grad():
        sigma = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / sigma
        conv.weight.mul_(scale.view(-1, 1, 1, 1))
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels,
                                                  device=conv.weight.device))
        conv.bias.copy_(scale * (conv.bias - bn.running_mean) + bn.bias)


class ConvVFI_UNet_ResBlock(nn.Module):
    """3-level U-Net with ResBlock (skip connections within each block).

    Same structure as ConvVFI_UNet but 2×Conv+ReLU replaced with ResBlock.
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 24):
        super().__init__()
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.enc1 = ResBlock(in_ch, ch1)
        self.down1 = nn.Sequential(
            nn.Conv2d(ch1, ch1, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = ResBlock(ch1, ch2)
        self.down2 = nn.Sequential(
            nn.Conv2d(ch2, ch2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.enc3 = ResBlock(ch2, ch3)

        self.up2 = nn.ConvTranspose2d(ch3, ch2, 2, stride=2)
        self.dec2 = ResBlock(ch2 * 2, ch2)
        self.up1 = nn.ConvTranspose2d(ch2, ch1, 2, stride=2)
        self.dec1 = ResBlock(ch1 * 2, ch1)
        self.out_conv = nn.Conv2d(ch1, 3, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class ConfidenceGate(nn.Module):
    """Lightweight per-pixel confidence map for frame blending.

    Predicts α ∈ [0,1]: blend = α·I0 + (1-α)·I1.
    Zero-initialized so initial α = sigmoid(0) = 0.5 (equal blend).
    """

    def __init__(self, in_ch: int = 6, hidden_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class ConvVFI_UNet_Gate(nn.Module):
    """U-Net with confidence-gated blending.

    Instead of fixed 0.5 blend, a lightweight gate predicts per-pixel α
    to weight the two prealigned frames. Residual is learned on top of
    the gated blend. Returns the final frame directly.
    """

    returns_frame = True

    def __init__(self, in_ch: int = 6, base_ch: int = 24):
        super().__init__()
        self.gate = ConfidenceGate(in_ch=in_ch, hidden_ch=16)
        self.unet = ConvVFI_UNet(in_ch=in_ch, base_ch=base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(x)  # (B, 1, H, W)
        gated_blend = alpha * x[:, :3] + (1 - alpha) * x[:, 3:6]
        residual = self.unet(x)
        return (gated_blend + residual).clamp(0, 1)


class ConvVFI_UNet_ResBlock_Gate(nn.Module):
    """U-Net with ResBlocks + confidence-gated blending.

    Combines both improvements: residual blocks for better feature
    extraction and confidence gate for adaptive frame weighting.
    Returns the final frame directly.
    """

    returns_frame = True

    def __init__(self, in_ch: int = 6, base_ch: int = 24):
        super().__init__()
        self.gate = ConfidenceGate(in_ch=in_ch, hidden_ch=16)
        self.unet = ConvVFI_UNet_ResBlock(in_ch=in_ch, base_ch=base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(x)  # (B, 1, H, W)
        gated_blend = alpha * x[:, :3] + (1 - alpha) * x[:, 3:6]
        residual = self.unet(x)
        return (gated_blend + residual).clamp(0, 1)


# ---------------------------------------------------------------------------
# NAFNet blocks (Phase 4 — MV ceiling test)
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and multiply: no learnable params."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet block: Norm → 1x1 → DWConv → SimpleGate → SCA → 1x1 → skip.

    From "Simple Baselines for Image Restoration" (ECCV 2022).
    No nonlinear activations — gate replaces ReLU/GELU.

    Args:
        norm_type: "ln" for LayerNorm2d (original), "bn" for BatchNorm2d
                   (HTP-friendly, fuses into Conv at inference).
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2,
                 norm_type: str = "ln"):
        super().__init__()
        dw_ch = c * dw_expand

        _norm = LayerNorm2d if norm_type == "ln" else nn.BatchNorm2d

        # Spatial mixing branch
        self.norm1 = _norm(c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)

        # Channel mixing branch (feed-forward)
        ffn_ch = c * ffn_expand
        self.norm2 = _norm(c)
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)

        # Learnable skip scaling (init=0 → identity at start)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        # Channel mixing
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        return x + y * self.gamma


class NAFNetVFI(nn.Module):
    """NAFNet encoder-decoder for VFI refinement (ceiling test).

    4-level encoder-decoder with NAFBlocks, skip connections,
    stride-2 conv down / PixelShuffle up.  Auto-pads input to
    multiple of 2^(num_levels) for arbitrary resolution support.
    """

    def __init__(
        self,
        in_ch: int = 6,
        width: int = 32,
        enc_blk_nums: tuple[int, ...] = (1, 1, 1, 28),
        middle_blk_num: int = 1,
        dec_blk_nums: tuple[int, ...] = (1, 1, 1, 1),
        norm_type: str = "ln",
    ):
        super().__init__()
        self.num_levels = len(enc_blk_nums)
        self.pad_multiple = 2 ** self.num_levels

        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch, norm_type=norm_type) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(ch, norm_type=norm_type) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch, norm_type=norm_type) for _ in range(num)])
            )

        # Zero-init output so initial residual = 0
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # Pad to multiple of 2^num_levels
        pad_h = (self.pad_multiple - H % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - W % self.pad_multiple) % self.pad_multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.intro(x)

        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.ending(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


def load_nafnet_pretrained(
    model: NAFNetVFI,
    pretrained_path: str,
) -> tuple[int, int]:
    """Load NAFNet pretrained weights, adapting intro conv from 3ch to in_ch.

    Returns (loaded_keys, skipped_keys) counts.
    """
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    pre_sd = ckpt["params"] if "params" in ckpt else ckpt

    model_sd = model.state_dict()
    new_sd = {}
    skipped = 0

    for k, v in pre_sd.items():
        if k not in model_sd:
            skipped += 1
            continue
        if k == "intro.weight" and v.shape[1] != model_sd[k].shape[1]:
            # Adapt 3ch → 6ch: average pretrained RGB weights, tile to in_ch
            # pretrained: (width, 3, 3, 3) → ours: (width, in_ch, 3, 3)
            in_ch = model_sd[k].shape[1]
            avg = v.mean(dim=1, keepdim=True)  # (width, 1, 3, 3)
            new_sd[k] = avg.expand(-1, in_ch, -1, -1).clone()
        elif v.shape == model_sd[k].shape:
            new_sd[k] = v
        else:
            skipped += 1

    model.load_state_dict(new_sd, strict=False)
    return len(new_sd), skipped


# ---------------------------------------------------------------------------
# Phase 6 — FlowRefineNet: pure Conv MV flow refinement + differentiable warp
# ---------------------------------------------------------------------------


class FlowRefineNet(nn.Module):
    """Small pure-Conv network that refines MV flow (residual learning).

    Input: normalized flow (B, 2, H, W) where flow = pixel_disp / (W, H).
    Output: refined flow (B, 2, H, W), same normalization.
    Zero-initialized last layer so initial output = input flow (no-op).
    """

    def __init__(self, mid_ch: int = 16, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(2, mid_ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(mid_ch, 2, 3, padding=1))
        self.net = nn.Sequential(*layers)
        # Zero init last layer for residual learning
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        return flow + self.net(flow)


class FlowRefineVFI(nn.Module):
    """Flow refinement + differentiable warp + residual correction.

    End-to-end trainable: gradients flow through F.grid_sample back to FlowRefineNet.
    Input: (B, 8, H, W) = [I0(3ch), I1(3ch), flow_norm(2ch)]
    Output: (B, 3, H, W) = predicted middle frame.

    The flow_norm is MV dense flow divided by (W, H), values ~[-0.1, 0.1].
    FlowRefineNet refines it, then we warp I0/I1 to midpoint and apply residual correction.
    """

    returns_frame = True  # handles warp + blend + residual internally

    def __init__(
        self,
        residual_cls: type[nn.Module] = ConvVFI_UNet,
        residual_kwargs: dict | None = None,
        refine_mid_ch: int = 16,
        refine_layers: int = 3,
    ):
        super().__init__()
        self.flow_refine = FlowRefineNet(refine_mid_ch, refine_layers)
        self.residual_net = residual_cls(**(residual_kwargs or {"in_ch": 6}))

    def _warp(self, img: torch.Tensor, flow_norm: torch.Tensor) -> torch.Tensor:
        """Warp image using normalized flow.

        Args:
            img: (B, 3, H, W) source image.
            flow_norm: (B, 2, H, W) flow normalized by (W, H).
                       flow_norm[:, 0] = pixel_dx / W, flow_norm[:, 1] = pixel_dy / H.
        Returns:
            (B, 3, H, W) warped image.
        """
        B, _, H, W = img.shape
        # Base grid: pixel coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=img.device, dtype=img.dtype),
            torch.arange(W, device=img.device, dtype=img.dtype),
            indexing="ij",
        )
        # Sample coordinates = base + pixel displacement
        sample_x = grid_x.unsqueeze(0) + flow_norm[:, 0] * W  # (B, H, W)
        sample_y = grid_y.unsqueeze(0) + flow_norm[:, 1] * H
        # Normalize to [-1, 1] for grid_sample(align_corners=True)
        sample_x = 2.0 * sample_x / (W - 1) - 1.0
        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        grid = torch.stack([sample_x, sample_y], dim=-1)  # (B, H, W, 2)
        return F.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        I0, I1, flow_norm = x[:, :3], x[:, 3:6], x[:, 6:8]
        # Refine flow (I0→I1 forward flow, normalized by W,H)
        refined_flow = self.flow_refine(flow_norm)
        # Warp to midpoint: I0 backward-sample by -flow/2, I1 forward-sample by +flow/2
        aligned_I0 = self._warp(I0, refined_flow * -0.5)
        aligned_I1 = self._warp(I1, refined_flow * 0.5)
        # Residual correction on prealigned pair
        inp = torch.cat([aligned_I0, aligned_I1], dim=1)  # (B, 6, H, W)
        residual = self.residual_net(inp)
        blend = (aligned_I0 + aligned_I1) * 0.5
        return (blend + residual).clamp(0, 1)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple[type[nn.Module], dict]] = {
    # Paper models: D-unet-v3b{s,m}-nomv (ANVIL-S/M), D-nafnet-nomv (NAFNet ceiling)
    # Ablation models: D-{tiny,mini,mid,unet-s,unet-l}-nomv (Table V capacity sweep)
    #                  A-small (Route A baseline, Table V)
    # All other entries are exploration history and not referenced in the paper.
    # FR-* models use F.grid_sample (flow refinement exploration, not NPU-deployable).
    #
    # Exp 2 — Route comparison
    "A-small":      (ConvVFI_Small, {"in_ch": 6}),
    "A-large":      (ConvVFI_Large, {"in_ch": 6}),
    "D-small":      (ConvVFI_Small, {"in_ch": 8}),
    "D-large":      (ConvVFI_Large, {"in_ch": 8}),
    "D-small-nomv": (ConvVFI_Small, {"in_ch": 6}),
    "D-large-nomv": (ConvVFI_Large, {"in_ch": 6}),
    # Exp 3 — Capacity sweep (all Route D with MV, in_ch=8)
    "D-tiny":   (ConvVFI_Plain, {"in_ch": 8, "channels": 8,  "n_layers": 4}),
    "D-mini":   (ConvVFI_Plain, {"in_ch": 8, "channels": 16, "n_layers": 6}),
    "D-mid":    (ConvVFI_Plain, {"in_ch": 8, "channels": 24, "n_layers": 8}),
    "D-unet-s": (ConvVFI_UNet,  {"in_ch": 8, "base_ch": 16}),
    "D-unet-l": (ConvVFI_UNet,  {"in_ch": 8, "base_ch": 24}),
    # Capacity sweep — D-nomv (production route, in_ch=6)
    "D-tiny-nomv":   (ConvVFI_Plain, {"in_ch": 6, "channels": 8,  "n_layers": 4}),
    "D-mini-nomv":   (ConvVFI_Plain, {"in_ch": 6, "channels": 16, "n_layers": 6}),
    "D-mid-nomv":    (ConvVFI_Plain, {"in_ch": 6, "channels": 24, "n_layers": 8}),
    "D-unet-s-nomv": (ConvVFI_UNet,  {"in_ch": 6, "base_ch": 16}),
    "D-unet-l-nomv": (ConvVFI_UNet,  {"in_ch": 6, "base_ch": 24}),
    # Phase 3 — Architecture ablations on D-unet-l-nomv
    "D-unet-l-nomv-resblock":      (ConvVFI_UNet_ResBlock,      {"in_ch": 6, "base_ch": 24}),
    "D-unet-l-nomv-gate":          (ConvVFI_UNet_Gate,          {"in_ch": 6, "base_ch": 24}),
    "D-unet-l-nomv-resblock-gate": (ConvVFI_UNet_ResBlock_Gate, {"in_ch": 6, "base_ch": 24}),
    # Phase 4 — MV ceiling test (NAFNet strong refinement)
    "D-nafnet-nomv": (NAFNetVFI, {
        "in_ch": 6, "width": 32,
        "enc_blk_nums": (1, 1, 1, 28), "middle_blk_num": 1,
        "dec_blk_nums": (1, 1, 1, 1),
    }),
    # Phase 5 — NAFNet-BN distillation targets (LN→BN for HTP deployment)
    "D-nafnet-bn-nomv": (NAFNetVFI, {
        "in_ch": 6, "width": 32,
        "enc_blk_nums": (1, 1, 1, 28), "middle_blk_num": 1,
        "dec_blk_nums": (1, 1, 1, 1),
        "norm_type": "bn",
    }),
    "D-nafnet-bn-s-nomv": (NAFNetVFI, {
        "in_ch": 6, "width": 16,
        "enc_blk_nums": (1, 1, 1, 8), "middle_blk_num": 1,
        "dec_blk_nums": (1, 1, 1, 1),
        "norm_type": "bn",
    }),
    # Phase 5b — UNet-v2: HTP-optimized DWBlock U-Net
    # DWConv+1×1+SCA blocks, 4-level U-Net, deep bottleneck, expand=2
    "D-unet-v2-s-nomv": (ConvVFI_UNetV2, {
        "in_ch": 6, "base_ch": 16, "bottleneck_blocks": 8, "expand": 2,
    }),
    "D-unet-v2-m-nomv": (ConvVFI_UNetV2, {
        "in_ch": 6, "base_ch": 24, "bottleneck_blocks": 8, "expand": 2,
    }),
    "D-unet-v2-l-nomv": (ConvVFI_UNetV2, {
        "in_ch": 6, "base_ch": 24, "bottleneck_blocks": 8, "expand": 2,
        "enc_blocks": (2, 2, 2, 2),
    }),
    # Phase 5 probe — same arch as bn-s but with LN, for HTP latency comparison
    "D-nafnet-ln-s-nomv": (NAFNetVFI, {
        "in_ch": 6, "width": 16,
        "enc_blk_nums": (1, 1, 1, 8), "middle_blk_num": 1,
        "dec_blk_nums": (1, 1, 1, 1),
        "norm_type": "ln",
    }),
    # Phase 5c — UNet-v3: INT8-native, standard Conv 3×3 + ResBlock
    # Designed from INT8 per-op profiling: Conv 3×3 is compute-bound (4.0x INT8 speedup),
    # DWConv/SCA are memory-bound (2.4x). ch_mults=(1,2,4,4) caps at 4× for efficiency.
    "D-unet-v3-s-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 20, "ch_mults": (1, 2, 4, 4),
        "enc_blocks": (1, 1, 1, 2), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 2,
    }),
    "D-unet-v3-m-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 24, "ch_mults": (1, 2, 4, 4),
        "enc_blocks": (1, 1, 1, 2), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 4,
    }),
    "D-unet-v3-l-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 24, "ch_mults": (1, 2, 4, 4),
        "enc_blocks": (1, 2, 2, 4), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 4,
    }),
    # Phase 5c — UNet-v3 optimized: asymmetric channels (narrow full-res, wide deep)
    "D-unet-v3xs-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 8, "ch_mults": (1, 2, 6, 6),
        "enc_blocks": (1, 1, 1, 1), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 4,
    }),
    # anvil-s candidates: deep channels also narrowed
    "D-unet-v3t-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 8, "ch_mults": (1, 2, 4, 4),
        "enc_blocks": (1, 1, 1, 1), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 4,
    }),
    "D-unet-v3tt-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 8, "ch_mults": (1, 2, 3, 3),
        "enc_blocks": (1, 1, 1, 1), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 4,
    }),
    "D-unet-v3s-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 16, "ch_mults": (1, 2, 4, 4),
        "enc_blocks": (1, 1, 1, 2), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 4,
    }),
    "D-unet-v3m-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 16, "ch_mults": (1, 2, 6, 6),
        "enc_blocks": (1, 1, 2, 2), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 8,
    }),
    "D-unet-v3l-nomv": (ConvVFI_UNetV3, {
        "in_ch": 6, "base_ch": 16, "ch_mults": (1, 2, 6, 6),
        "enc_blocks": (1, 1, 2, 4), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 8,
    }),
    # Phase 5d — UNet-v3b: v3 + additive skip + BN option
    # Profiling-driven: removes Concat+proj (saves ~5-8%), BN fuses to zero cost
    "D-unet-v3bs-nomv": (ConvVFI_UNetV3b, {
        "in_ch": 6, "base_ch": 16, "ch_mults": (1, 2, 4, 4),
        "enc_blocks": (1, 1, 1, 2), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 4, "norm": "bn",
    }),
    "D-unet-v3bm-nomv": (ConvVFI_UNetV3b, {
        "in_ch": 6, "base_ch": 16, "ch_mults": (1, 2, 6, 6),
        "enc_blocks": (1, 1, 2, 2), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 8, "norm": "bn",
    }),
    # Same configs without BN for ablation
    "D-unet-v3bs-nobn-nomv": (ConvVFI_UNetV3b, {
        "in_ch": 6, "base_ch": 16, "ch_mults": (1, 2, 4, 4),
        "enc_blocks": (1, 1, 1, 2), "dec_blocks": (1, 1, 1, 1),
        "bottleneck_blocks": 4, "norm": "none",
    }),
    # Phase 6 — FlowRefineNet: learned MV flow refinement + differentiable warp
    "FR-unet-l": (FlowRefineVFI, {
        "residual_cls": ConvVFI_UNet,
        "residual_kwargs": {"in_ch": 6, "base_ch": 24},
        "refine_mid_ch": 16, "refine_layers": 3,
    }),
}


def build_model(model_id: str) -> nn.Module:
    """Instantiate a model by its registry ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    cls, kwargs = MODEL_REGISTRY[model_id]
    return cls(**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_route(model_id: str) -> str:
    """Infer data route from model ID.

    FR-*     → 'FR'     (raw frames + MV flow, 8ch, for FlowRefineNet)
    A-*      → 'A'      (raw frames, 6ch)
    D-*-nomv → 'D-nomv' (prealigned, no MV, 6ch)
    D-*      → 'D'      (prealigned + MV, 8ch)
    """
    if model_id.startswith("FR-"):
        return "FR"
    if model_id.startswith("A-"):
        return "A"
    if "-nomv" in model_id:
        return "D-nomv"
    return "D"
