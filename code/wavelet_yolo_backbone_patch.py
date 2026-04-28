"""
wavelet_yolo_backbone_patch.py

Wavelet modules + YOLO backbone patch utilities with toggles:
- Input DWT
- Wavelet pooling
- Wavelet-based channel attention
- WaveletConv2d (DWT -> conv -> upsample)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1) Basic DWT block (Haar, fixed filters, GPU-friendly)
# =========================================================

class WaveletDWT2D(nn.Module):
    """
    Simple 2D DWT using fixed Haar filters implemented as grouped convs.

    Input:  [B, C, H, W]
    Output: [B, 4*C, H/2, W/2]  (LL, LH, HL, HH stacked along channel dim)
    """
    def __init__(self):
        super().__init__()
        # Haar filters (2x2) normalized
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5,  0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5,  0.5],
                           [-0.5,  0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]], dtype=torch.float32)

        # We’ll build filters dynamically per channel using these base kernels.
        base_kernels = torch.stack([ll, lh, hl, hh], dim=0)  # [4, 2, 2]
        self.register_buffer("base_kernels", base_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Pad if odd size to keep DWT shape valid
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            _, _, h, w = x.shape

        # Build grouped conv weights of shape (4C, 1, 2, 2) with groups=C
        kernels = self.base_kernels.to(x.device)  # [4, 2, 2]
        # Expand to (4C, 1, 2, 2)
        kernels = kernels.view(4, 1, 2, 2).repeat(c, 1, 1, 1)  # [4C,1,2,2]

        y = F.conv2d(x, kernels, stride=2, padding=0, groups=c)  # [B,4C,H/2,W/2]
        return y

    @staticmethod
    def split_subbands(y: torch.Tensor) -> tuple:
        """
        Split [B, 4C, H, W] into (LL, LH, HL, HH), each [B, C, H, W].
        """
        b, ch, h, w = y.shape
        assert ch % 4 == 0, "Channel dim must be divisible by 4"
        c = ch // 4
        LL, LH, HL, HH = torch.split(y, c, dim=1)
        return LL, LH, HL, HH


# =========================================================
# 2) Wavelet pooling (drop-in replacement for MaxPool2d)
# =========================================================

class WaveletPool2d(nn.Module):
    """
    Wavelet-based pooling:
      - Run DWT
      - Keep LL (low-frequency) subband as pooled output

    Acts like a learned “smart average+edge-aware” pooling step.
    Input:  [B, C, H, W]
    Output: [B, C, H/2, W/2]
    """
    def __init__(self):
        super().__init__()
        self.dwt = WaveletDWT2D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwt(x)                 # [B, 4C, H/2, W/2]
        LL, _, _, _ = self.dwt.split_subbands(y)
        return LL


# =========================================================
# 3) Wavelet-based channel attention
#     - Uses DWT to derive frequency-aware channel weights
#     - Applies weights to original feature map (no shape change)
# =========================================================

class WaveletChannelAttention(nn.Module):
    """
    Wavelet-based channel attention.

    Idea:
      1. DWT(x) -> LL, LH, HL, HH  (each [B, C, H/2, W/2])
      2. Global average pooling per subband
      3. Concatenate stats and pass through small MLP
      4. Use result to weight original x channels (like SE/ECA)

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.dwt = WaveletDWT2D()

        hidden = max(channels // reduction, 4)

        # MLP over concatenated [LL,LH,HL,HH] pooled stats: 4*C
        self.fc1 = nn.Conv1d(4, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden, 1, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        y = self.dwt(x)                      # [B, 4C, H/2, W/2]
        LL, LH, HL, HH = self.dwt.split_subbands(y)

        # Global average pooling per subband -> [B, C]
        def gap(z: torch.Tensor) -> torch.Tensor:
            return F.adaptive_avg_pool2d(z, 1).view(b, c)

        stats = torch.stack([
            gap(LL), gap(LH), gap(HL), gap(HH)
        ], dim=1)                            # [B, 4, C]

        # MLP across 4-subband dimension -> per-channel weight
        z = self.fc1(stats)                  # [B, hidden, C]
        z = self.act(z)
        z = self.fc2(z)                      # [B, 1, C]
        z = z.view(b, c)                     # [B, C]
        w = self.sigmoid(z).view(b, c, 1, 1)

        return x * w


# =========================================================
# 4) WaveletConv2d (DWT -> conv in wavelet space -> upsample)
#     - Drop-in replacement for Conv2d (same input/output shape)
# =========================================================

class WaveletConv2d(nn.Module):
    """
    Wavelet-aware Conv2d.

    Workflow:
      1. DWT(x) -> [B, 4*C_in, H/2, W/2]
      2. Conv in wavelet space
      3. Upsample back to original resolution
      4. 1x1 Conv to out_channels

    This is NOT a mathematically exact inverse, but acts as a
    frequency-aware conv layer with same interface as Conv2d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        wavelet_channels_factor: int = 4,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        activation: Optional[nn.Module] = nn.SiLU(inplace=True),
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwt = WaveletDWT2D()
        self.wavelet_channels_factor = wavelet_channels_factor

        mid_channels = wavelet_channels_factor * in_channels  # 4*C_in
        self.conv_wavelet = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=1,              # DWT already downsamples
            padding=padding,
            bias=bias,
        )

        # Reprojection back to out_channels after upsampling
        self.conv_out = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = activation

        # Optionally support emulating stride > 1 by extra pooling
        self.post_stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        y = self.dwt(x)                      # [B, 4*C, H/2, W/2]
        y = self.conv_wavelet(y)             # [B, 4*C, H/2, W/2]

        # Upsample to original H,W
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        y = self.conv_out(y)                 # [B, out_channels, H, W]

        if self.post_stride > 1:
            # emulate stride by avgpool
            y = F.avg_pool2d(y, kernel_size=self.post_stride, stride=self.post_stride)

        if self.norm is not None:
            y = self.norm(y)
        if self.act is not None:
            y = self.act(y)
        return y


# =========================================================
# 5) Toggle config
# =========================================================

@dataclass
class WaveletToggleConfig:
    """
    Global config for enabling wavelet modules.

    You can keep this separate from your existing backbone
    patch config or merge it into your existing dataclass.
    """
    replace_stem: bool = False              # Only patch/replace the input stem if True
    enable_input_dwt: bool = False          # Replace first conv with WaveletConv2d
    enable_wavelet_pool: bool = False       # Replace MaxPool2d with WaveletPool2d
    enable_wavelet_attention: bool = False  # Add WaveletChannelAttention after selected convs
    enable_wavelet_conv: bool = False       # Replace selected Conv2d with WaveletConv2d

    # Where / how aggressively to apply
    attention_reduction: int = 16
    wavelet_conv_on_stem_only: bool = True  # If True, only first conv; else on selected list

    # Safety: only patch/replace the input stem if explicitly enabled.
    replace_stem: bool = False

    # Names or indices of modules you want to target
    target_conv_module_names: Optional[Sequence[str]] = None
    target_attention_module_names: Optional[Sequence[str]] = None
    # If you have Ultralytics 'Conv' blocks etc., you can
    # target those layer names / indices with these lists.



class ResidualWaveletFromConv(nn.Module):
    """
    Identity-safe residual wavelet wrapper for an existing Conv2d:

        y = conv(x) + alpha * wavelet_branch(x)

    - Works for any input channel count C
    - Starts as identity when alpha=0 (pretrained weights remain valid)
    - alpha can be ramped during training (see trainer callback)
    """

    def __init__(self, conv: nn.Conv2d, alpha_init: float = 0.0):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"ResidualWaveletFromConv expects nn.Conv2d, got {type(conv)}")

        self.conv = conv  # keep pretrained base path

        self.dwt = WaveletDWT2D()
        cin = conv.in_channels
        cout = conv.out_channels

        # Wavelet branch operates on DWT features (4*Cin channels)
        self.conv_wavelet = nn.Conv2d(
            4 * cin,
            4 * cin,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv_out = nn.Conv2d(4 * cin, cout, kernel_size=1, bias=True)

        # alpha is a buffer so it's saved with state_dict but not trained by optimizer
        self.register_buffer("alpha", torch.tensor(float(alpha_init)))

        # Make wavelet branch quiet initially too
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def set_alpha(self, a: float):
        self.alpha.fill_(float(a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.conv(x)

        y = self.dwt(x)  # [B, 4C, H/2, W/2]
        y = self.conv_wavelet(y)

        # Match base spatial size (handles stride=2 stems cleanly)
        y = F.interpolate(y, size=y_base.shape[-2:], mode="bilinear", align_corners=False)
        y = self.conv_out(y)

        return y_base + self.alpha * y


def patch_input_with_residual_wavelet(model: nn.Module, alpha_init: float = 0.0, inject_stage: int = 1):
    """
    Wrap a selected nn.Conv2d in the model with ResidualWaveletFromConv.

    inject_stage controls WHICH downsampling stage to wrap, counted by Conv2d with stride=2:
      - inject_stage=1 -> first stride-2 conv (typically the stem)
      - inject_stage=2 -> second stride-2 conv (typically stage2 / early backbone)
      - inject_stage=3 -> third stride-2 conv (deeper)
    This preserves any prior multispectral first-conv remapping.
    """
    target_conv = None
    target_name = None

    # Prefer selecting by downsample stage (stride=2 conv count)
    down = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            s = m.stride[0] if isinstance(m.stride, tuple) else int(m.stride)
            if s == 2:
                down += 1
                if down == int(inject_stage):
                    target_conv, target_name = m, name
                    break

    # Fallback: if stage not found, wrap the very first conv (safe default)
    if target_conv is None:
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                target_conv, target_name = m, name
                break

    if target_conv is None:
        print("[WaveletPatch] No Conv2d found to wrap.")
        return model

    wrapped = ResidualWaveletFromConv(target_conv, alpha_init=alpha_init)

    parts = target_name.split(".")
    root = model
    for p in parts[:-1]:
        root = getattr(root, p)
    setattr(root, parts[-1], wrapped)

    print(f"[WaveletPatch] Wrapped Conv2d '{target_name}' (inject_stage={inject_stage}) with ResidualWaveletFromConv (alpha_init={alpha_init}).")
    return model

    wrapped = ResidualWaveletFromConv(first_conv, alpha_init=alpha_init)

    parts = first_name.split(".")
    root = model
    for p in parts[:-1]:
        root = getattr(root, p)
    setattr(root, parts[-1], wrapped)

    print(f"[WaveletPatch] Wrapped first Conv2d '{first_name}' with ResidualWaveletFromConv (alpha_init={alpha_init}).")
    return model

# =========================================================
# 6) Backbone patch helpers
#     NOTE: These are generic PyTorch; adapt to your
#           Ultralytics YOLO module layout as needed.
# =========================================================

def _iter_named_modules(model: nn.Module):
    """
    Yield (full_name, module) pairs for all submodules.
    """
    for name, module in model.named_modules():
        yield name, module


def patch_input_with_wavelet_conv(
    model: nn.Module,
    cfg: WaveletToggleConfig,
    norm_layer_factory: Optional[Callable[[int], nn.Module]] = None,
):
    """
    Replace *first* Conv2d-like layer with WaveletConv2d.

    For Ultralytics YOLO, you might adapt this to target model.model[0].conv
    or your custom ConvWithAttention wrapper. Here we search for the
    first nn.Conv2d by default.
    """
    if not cfg.enable_input_dwt and not cfg.enable_wavelet_conv:
        return model

    first_conv: Optional[nn.Conv2d] = None
    first_name: Optional[str] = None

    for name, m in _iter_named_modules(model):
        if isinstance(m, nn.Conv2d):
            first_conv = m
            first_name = name
            break

    if first_conv is None:
        print("[WaveletPatch] No Conv2d layer found for input patch.")
        return model

    in_ch = first_conv.in_channels
    out_ch = first_conv.out_channels
    ks = first_conv.kernel_size[0]
    stride = first_conv.stride[0]
    pad = first_conv.padding[0]
    bias = first_conv.bias is not None

    # Build WaveletConv2d with matching IO
    def build_norm(c):
        return norm_layer_factory(c) if norm_layer_factory is not None else None

    wconv = WaveletConv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=ks,
        stride=stride,
        padding=pad,
        bias=bias,
        norm_layer=build_norm,
        activation=nn.SiLU(inplace=True),
    )

    # Recursively replace the module by name
    def _replace(root: nn.Module, target_name: str, new_module: nn.Module):
        parts = target_name.split(".")
        for p in parts[:-1]:
            root = getattr(root, p)
        setattr(root, parts[-1], new_module)

    _replace(model, first_name, wconv)
    print(f"[WaveletPatch] Replaced first Conv2d '{first_name}' with WaveletConv2d.")
    return model


def replace_maxpool_with_wavelet_pool(model: nn.Module, cfg: WaveletToggleConfig):
    """
    Replace all nn.MaxPool2d with WaveletPool2d (if enabled).
    You can customise this to only replace certain ones.
    """
    if not cfg.enable_wavelet_pool:
        return model

    for name, m in list(_iter_named_modules(model)):
        if isinstance(m, nn.MaxPool2d):
            # Replace this MaxPool2d
            def _replace(root: nn.Module, target_name: str, new_module: nn.Module):
                parts = target_name.split(".")
                for p in parts[:-1]:
                    root = getattr(root, p)
                setattr(root, parts[-1], new_module)

            _replace(model, name, WaveletPool2d())
            print(f"[WaveletPatch] Replaced MaxPool2d '{name}' with WaveletPool2d.")

    return model


def add_wavelet_attention_after_convs(
    model: nn.Module,
    cfg: WaveletToggleConfig,
    conv_filter: Optional[Callable[[str, nn.Module], bool]] = None,
):
    """
    Wrap selected Conv2d-like layers with WaveletChannelAttention.

    For generic PyTorch:
      - If layer is Conv2d, we replace it with Sequential(Conv2d, WaveletChannelAttention)
    For Ultralytics 'Conv' blocks:
      - You may want to modify this to attach .attention = WaveletChannelAttention(...)
    """
    if not cfg.enable_wavelet_attention:
        return model

    if conv_filter is None:
        # Default: all Conv2d layers if no filter provided
        def conv_filter(name, module):
            return isinstance(module, nn.Conv2d)

    for name, m in list(_iter_named_modules(model)):
        if conv_filter(name, m):
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels
            else:
                # Fallback; user can override conv_filter and logic
                continue

            att = WaveletChannelAttention(channels, reduction=cfg.attention_reduction)
            seq = nn.Sequential(m, att)

            # Replace in parent
            def _replace(root: nn.Module, target_name: str, new_module: nn.Module):
                parts = target_name.split(".")
                for p in parts[:-1]:
                    root = getattr(root, p)
                setattr(root, parts[-1], new_module)

            _replace(model, name, seq)
            print(f"[WaveletPatch] Added WaveletChannelAttention after '{name}'.")

    return model


def replace_selected_convs_with_wavelet_conv(
    model: nn.Module,
    cfg: WaveletToggleConfig,
    norm_layer_factory: Optional[Callable[[int], nn.Module]] = None,
):
    """
    Replace selected Conv2d layers with WaveletConv2d.

    - If wavelet_conv_on_stem_only=True, this is effectively a no-op
      because patch_input_with_wavelet_conv() already handles first conv.
    - If False, we use cfg.target_conv_module_names as a whitelist.
    """
    if not cfg.enable_wavelet_conv:
        return model
    if cfg.wavelet_conv_on_stem_only:
        # Already handled by patch_input_with_wavelet_conv
        return model

    targets = set(cfg.target_conv_module_names or [])
    if not targets:
        print("[WaveletPatch] No target_conv_module_names provided, skipping WaveletConv2d replacements.")
        return model

    def build_norm(c):
        return norm_layer_factory(c) if norm_layer_factory is not None else None

    for name, m in list(_iter_named_modules(model)):
        if name in targets and isinstance(m, nn.Conv2d):
            in_ch = m.in_channels
            out_ch = m.out_channels
            ks = m.kernel_size[0]
            stride = m.stride[0]
            pad = m.padding[0]
            bias = m.bias is not None

            wconv = WaveletConv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=ks,
                stride=stride,
                padding=pad,
                bias=bias,
                norm_layer=build_norm,
                activation=nn.SiLU(inplace=True),
            )

            # Replace
            def _replace(root: nn.Module, target_name: str, new_module: nn.Module):
                parts = target_name.split(".")
                for p in parts[:-1]:
                    root = getattr(root, p)
                setattr(root, parts[-1], new_module)

            _replace(model, name, wconv)
            print(f"[WaveletPatch] Replaced Conv2d '{name}' with WaveletConv2d.")

    return model




def warn_if_stem_is_wavelet_conv(model: nn.Module):
    """Debug helper: warn if a WaveletConv2d exists in the model (often means stem was replaced)."""
    for name, m in model.named_modules():
        if isinstance(m, WaveletConv2d):
            print(f"[WARN] WaveletConv2d found at '{name}' — stem may have been replaced.")
            break
# =========================================================
# 7) Convenience: one-call patch
# =========================================================

def apply_wavelet_patches(
    model: nn.Module,
    cfg: WaveletToggleConfig,
    norm_layer_factory: Optional[Callable[[int], nn.Module]] = None,
    conv_attention_filter: Optional[Callable[[str, nn.Module], bool]] = None,
) -> nn.Module:
    """
    Apply all wavelet patches in a single call.

    In your main YOLO script, after building the model:

        from wavelet_yolo_backbone_patch import (
            WaveletToggleConfig, apply_wavelet_patches
        )

        cfg = WaveletToggleConfig(
            enable_input_dwt=True,
            enable_wavelet_pool=True,
            enable_wavelet_attention=True,
            enable_wavelet_conv=False,
            wavelet_conv_on_stem_only=True,
        )

        model = apply_wavelet_patches(model, cfg)

    Then proceed with training as usual.
    """

    # Input stem patching (dangerous): only do this if explicitly requested.
    if bool(getattr(cfg, "replace_stem", False)):
        model = patch_input_with_wavelet_conv(model, cfg, norm_layer_factory)
    else:
        # Ensure we do not accidentally replace the stem conv.
        pass


    # Wavelet pooling
    model = replace_maxpool_with_wavelet_pool(model, cfg)

    # Extra WaveletConv replacements
    model = replace_selected_convs_with_wavelet_conv(model, cfg, norm_layer_factory)

    # Wavelet attention
    model = add_wavelet_attention_after_convs(
        model, cfg, conv_filter=conv_attention_filter
    )

    return model



# ======================================================================================
# Compatibility helpers for training scripts (WNNDetectionTrainer / experiment runner)
# Adds missing symbols that some scripts import.
# ======================================================================================

import torch
import torch.nn as nn

def tag_backbone_and_head_layers(model: nn.Module, backbone_last_idx: int = 8) -> None:
    """
    Tag Ultralytics-style model layers with boolean flags:
      - module._is_backbone
      - module._is_head
    Assumes `model.model` is an nn.ModuleList (Ultralytics convention).
    Safe no-op if not present.
    """
    layers = getattr(model, "model", None)
    if layers is None:
        return
    try:
        n = len(layers)
    except Exception:
        return
    for i, m in enumerate(layers):
        try:
            setattr(m, "_is_backbone", bool(i <= backbone_last_idx))
            setattr(m, "_is_head", bool(i > backbone_last_idx))
        except Exception:
            pass

def mark_wnn_modules(model: nn.Module) -> None:
    """
    Mark wavelet/WNN modules so the optimizer can group parameters reliably.
    Sets: module._is_wnn = True for Wavelet* modules.
    """
    WNN_TYPES = tuple(t for t in [
        globals().get("WaveletConv2d", None),
        globals().get("WaveletChannelAttention", None),
        globals().get("WaveletPool2d", None),
        globals().get("WaveletDWT2D", None),
    ] if t is not None)
    if not WNN_TYPES:
        return
    for m in model.modules():
        if isinstance(m, WNN_TYPES):
            try:
                setattr(m, "_is_wnn", True)
            except Exception:
                pass

def replace_first_conv_with_multispectral_band_aware(model: nn.Module, band_names) -> nn.Module:
    """
    Replace the first nn.Conv2d in the model to accept len(band_names) input channels.
    Copies pretrained RGB weights into the first 3 channels (assumes order R,G,B),
    and initializes extra channels with the mean of RGB kernels (stable default).

    Returns the (possibly) modified model.
    """
    if band_names is None:
        return model
    if isinstance(band_names, str):
        band_names = [b.strip() for b in band_names.split(",") if b.strip()]
    in_ch = int(len(band_names))
    if in_ch <= 0:
        return model

    # find first conv and its parent
    first = None
    first_parent = None
    first_name = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first = module
            # locate parent module by walking attributes
            # (Ultralytics uses nested ModuleList; this generic method works for most cases)
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            first_parent = parent
            first_name = parts[-1]
            break

    if first is None:
        return model

    if first.in_channels == in_ch:
        return model

    # build replacement conv with same hyperparams
    new_conv = nn.Conv2d(
        in_channels=in_ch,
        out_channels=first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        dilation=first.dilation,
        groups=first.groups if first.groups == 1 else 1,  # be conservative
        bias=(first.bias is not None),
        padding_mode=first.padding_mode,
    )

    with torch.no_grad():
        # copy existing weights where possible
        w_old = first.weight.data  # [out, in, k, k]
        w_new = new_conv.weight.data
        oc, ic_old, kh, kw = w_old.shape
        ic_copy = min(ic_old, in_ch)
        w_new[:, :ic_copy, :, :] = w_old[:, :ic_copy, :, :]
        if in_ch > ic_old:
            # init extra channels as mean of first 3 (or all) channels
            ref = w_old[:, :min(3, ic_old), :, :].mean(dim=1, keepdim=True)
            for c in range(ic_old, in_ch):
                w_new[:, c:c+1, :, :] = ref
        if first.bias is not None and new_conv.bias is not None:
            new_conv.bias.data.copy_(first.bias.data)

    # assign into parent
    if first_parent is not None and first_name is not None:
        if first_name.isdigit():
            first_parent[int(first_name)] = new_conv
        else:
            setattr(first_parent, first_name, new_conv)

    return model

# Backwards-compatible aliases (older scripts import these names)
WaveletConv = globals().get("WaveletConv2d", None)