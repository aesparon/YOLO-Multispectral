import torch
import torch.nn as nn
import torch.nn.functional as F

# Ultralytics Conv wrapper (YOLOv8 / YOLO11)
from ultralytics.nn.modules.conv import Conv as YConv


# ---------------------------------------------------------------------
# Basic SpectralConv stub
# ---------------------------------------------------------------------
class SpectralConv(nn.Module):
    """
    Simple spectral-style conv stub.

    For now this is just a standard Conv2d (we still call it 'spectral'
    so it plugs into your existing API). We copy the pretrained Conv
    weights into .combine to keep behaviour close to the original Conv.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.combine = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.combine(x)


# ---------------------------------------------------------------------
# CBAM: Channel + Spatial Attention
# ---------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out
        out = torch.sigmoid(out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(x_cat))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------------------------------------------------------
# ECA: Efficient Channel Attention
# ---------------------------------------------------------------------
class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, 1, c)
        y = self.conv1d(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y


# ---------------------------------------------------------------------
# DropBlock2D (simple variant)
# ---------------------------------------------------------------------
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob: float = 0.1, block_size: int = 7) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x

        b, c, h, w = x.size()
        gamma = (
            self.drop_prob
            * (h * w)
            / ((self.block_size**2) * (h - self.block_size + 1) * (w - self.block_size + 1) + 1e-8)
        )

        mask = (torch.rand(b, c, h - self.block_size + 1, w - self.block_size + 1, device=x.device) < gamma).float()
        mask = F.pad(mask, [self.block_size // 2] * 4)
        mask = 1 - F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)

        keep_ratio = mask.numel() / (mask.sum() + 1e-8)
        return x * mask * keep_ratio


# ---------------------------------------------------------------------
# Wrapper module to avoid closures (pickle-safe)
# ---------------------------------------------------------------------
# class ConvWithExtras(nn.Module):
#     """
#     Wraps a YOLO Conv block (YConv) with optional SpectralConv and
#     attention stack (CBAM/ECA/DropBlock).

#     forward(x):
#       x -> [SpectralConv?] -> ConvBlock(x) -> [Attn?]
#     """

#     def __init__(
#         self,
#         conv_block: YConv,
#         spectral: nn.Module | None = None,
#         attn: nn.Module | None = None,
#     ) -> None:
#         super().__init__()
#         self.conv_block = conv_block
#         self.spectral = spectral
#         self.attn = attn

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.spectral is not None:
#             x = self.spectral(x)
#         x = self.conv_block(x)
#         if self.attn is not None:
#             x = self.attn(x)
#         return x


class ConvWithExtras(nn.Module):
    """
    Wraps a YOLO Conv block (YConv) with optional SpectralConv and
    attention stack (CBAM/ECA/DropBlock), while preserving parameter
    names like 'model.0.conv.weight'.

    We copy the original Conv's `conv`, `bn`, and `act` attributes so
    the state_dict still has keys:
      - ...conv.weight
      - ...bn.weight
      - etc.
    """

    def __init__(
        self,
        base: YConv,
        spectral: nn.Module | None = None,
        attn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        # Reuse original conv / bn / act modules directly
        self.conv = base.conv
        self.bn = getattr(base, "bn", None)
        self.act = getattr(base, "act", nn.Identity())

        # New extras
        self.spectral = spectral
        self.attn = attn

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # Optional spectral pre-processing
    #     if self.spectral is not None:
    #         x = self.spectral(x)



    #     # Original Conv forward: conv -> bn -> act
    #     x = self.conv(x)


    #     if self.bn is not None:
    #         x = self.bn(x)
    #     if self.act is not None:
    #         x = self.act(x)

    #     # Optional attention / DropBlock on Conv outputs
    #     if self.attn is not None:
    #         x = self.attn(x)

    #     return x

def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Spectral replaces the conv (must be drop-in compatible)
    if self.spectral is not None:
        x = self.spectral(x)
    else:
        x = self.conv(x)

    if self.bn is not None:
        x = self.bn(x)
    if self.act is not None:
        x = self.act(x)

    # Optional attention / DropBlock on Conv outputs
    if self.attn is not None:
        x = self.attn(x)

    return x
# ---------------------------------------------------------------------
# Conv-level backbone patcher
# ---------------------------------------------------------------------
# def patch_backbone_with_attention(
#     model_nn: nn.Module,
#     use_cbam: bool = False,
#     use_eca: bool = False,
#     use_spectral: bool = False,
#     use_dropblock: bool = False,
#     drop_prob: float = 0.1,
#     use_groupnorm: bool = False,
#     gn_groups: int = 4,
# ) -> None:
def patch_backbone_with_attention(
    model_nn: nn.Module,
    use_cbam: bool = False,
    use_eca: bool = False,
    use_spectral: bool = False,
    use_dropblock: bool = False,
    drop_prob: float = 0.1,
    use_groupnorm: bool = False,
    gn_groups: int = 4,
    *,
    skip_head: bool = True,  # ✅ new (default safe)
) -> None:

    """
    Conv-level backbone patcher.

    - Optional BN -> GN replacement (use_groupnorm=True).
    - Optional SpectralConv pre-processing for the FIRST Conv.
    - Optional CBAM / ECA / DropBlock applied to Conv outputs.

    Implementation:
      * Operates directly on Ultralytics Conv blocks (YConv).
      * Replaces each YConv child with ConvWithExtras(conv, spectral?, attn?),
        which is top-level and pickle-safe.
    """
    print("\n🔧 Patching YOLO backbone (Conv-level)")

    # ---------------------- BN -> GN (optional) ----------------------
    if use_groupnorm:
        def norm_fn(channels: int) -> nn.Module:
            return nn.GroupNorm(gn_groups, channels)

        def replace_bn_with_gn(m: nn.Module):
            for name, child in list(m.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(m, name, norm_fn(child.num_features))
                    print(f"🔄 Replaced BatchNorm2d with GroupNorm({gn_groups}) at: {name}")
                else:
                    replace_bn_with_gn(child)

        replace_bn_with_gn(model_nn)
        print(f"✅ Finished BN → GN(g={gn_groups}) replacement")

    # ---------------------- Conv-level patching ----------------------
    spectral_used = False

    def recurse_patch(parent: nn.Module):
        nonlocal spectral_used

        for name, child in list(parent.named_children()):
            # If this child is a YOLO Conv block, consider patching it
            if isinstance(child, YConv) and isinstance(child.conv, nn.Conv2d):
                conv = child.conv
                c_in = conv.in_channels
                c_out = conv.out_channels

                spectral_module = None
                attn_blocks = []

                # ---- SpectralConv (first Conv only, if requested) ----
                if use_spectral and not spectral_used:
                    spectral_module = SpectralConv(
                        c_in,
                        c_out,
                        kernel_size=conv.kernel_size[0],
                        stride=conv.stride[0],
                        padding=conv.padding[0],
                        bias=(conv.bias is not None),
                    )
                    with torch.no_grad():
                        if spectral_module.combine.weight.shape == conv.weight.shape:
                            spectral_module.combine.weight.copy_(conv.weight.data)
                            print(f"🔁 SpectralConv: copied weights from Conv (c_in={c_in}, c_out={c_out})")
                    spectral_used = True
                    print(f"🔁 Attached SpectralConv before Conv (c_out={c_out})")

                # ---- CBAM / ECA / DropBlock --------------------------
                if use_cbam:
                    attn_blocks.append(CBAM(c_out))
                if use_eca:
                    attn_blocks.append(ECA(c_out))
                if use_dropblock and drop_prob > 0.0:
                    attn_blocks.append(DropBlock2D(drop_prob=drop_prob))

                if spectral_module is None and not attn_blocks:
                    # Nothing to do for this Conv
                    continue

                attn_seq = nn.Sequential(*attn_blocks) if attn_blocks else None

                # wrapped = ConvWithExtras(
                #     conv_block=child,
                #     spectral=spectral_module,
                #     attn=attn_seq,
                # )
                # setattr(parent, name, wrapped)

                # wrapped = ConvWithExtras(
                #     base=child,
                #     spectral=spectral_module,
                #     attn=attn_seq,
                # )
                # setattr(parent, name, wrapped)

                wrapped = ConvWithExtras(
                    base=child,
                    spectral=spectral_module,
                    attn=attn_seq,
                )

                # ✅ CRITICAL: preserve Ultralytics graph attributes (m.f, m.i, m.type, m.np)
                for attr in ("i", "f", "type", "np"):
                    if hasattr(child, attr):
                        setattr(wrapped, attr, getattr(child, attr))

                setattr(parent, name, wrapped)


                print(
                    f"✨ Patched Conv block at '{name}' (c_in={c_in}, c_out={c_out}) | "
                    f"spectral={spectral_module is not None}, attn={bool(attn_blocks)}"
                )

            else:
                # Recurse into children
                recurse_patch(child)

    # recurse_patch(model_nn)
    # print("✅ YOLO backbone patching complete\n")
    # ---------------------- Apply patch ----------------------
    # ✅ SAFETY: do NOT patch the final Segment head (breaks masks).
    # Most Ultralytics models store the graph in model_nn.model (nn.Sequential).
    if skip_head and hasattr(model_nn, "model") and isinstance(getattr(model_nn, "model"), nn.Sequential):
        seq = getattr(model_nn, "model")
        n = len(seq)
        for i in range(n - 1):  # ✅ skip last module (head)
            child = seq[i]
            # patch direct Conv blocks at this level
            if isinstance(child, YConv) and isinstance(child.conv, nn.Conv2d):
                # reuse the same logic by wrapping a tiny parent container:
                tmp_parent = nn.Module()
                setattr(tmp_parent, "x", child)
                recurse_patch(tmp_parent)
                seq[i] = getattr(tmp_parent, "x")
            else:
                recurse_patch(child)
        print("✅ YOLO backbone patching complete (head skipped)\n")
    else:
        recurse_patch(model_nn)
        print("✅ YOLO backbone patching complete\n")
